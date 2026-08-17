"""Source adapters: ABC, normalization, dedup, and fetch orchestrator."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from jobscout.db import find_by_dedup_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawPosting:
    """Intermediate representation returned by source adapters."""

    title: str
    source: str                    # adapter name, e.g. "wttj"
    company: str | None = None
    url: str | None = None
    description: str | None = None
    location: str | None = None
    country: str | None = None     # ISO 3166-1 alpha-2
    language: str | None = None    # posting language (en/fr/de)
    seniority: str | None = None
    posted_at: str | None = None
    source_id: str | None = None
    raw_data: dict | None = None


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class SourceAdapter(ABC):
    """Base class for all source adapters."""

    name: str

    @abstractmethod
    def fetch(self, source_config: dict) -> list[RawPosting]:
        """Fetch postings from this source. May raise on network errors."""
        ...


# ---------------------------------------------------------------------------
# Org-type classification
# ---------------------------------------------------------------------------

# Keywords are checked case-insensitively unless noted.

_LAB_COMPANY_KEYWORDS = {
    # Generic
    "university", "université", "universität", "universidad", "universiteit",
    "institut", "institute",
    "research center", "research centre", "laboratory", "laboratoire",
    "national lab", "academic", "faculty",
    # French institutions
    "cnrs", "inria", "inrae", "cea", "inserm", "ird", "cirad",
    "sciences po", "école polytechnique", "école normale",
    # German institutions
    "max planck", "fraunhofer", "helmholtz", "leibniz",
    # Other notable
    "epfl", "eth zurich", "eth zürich", "mit ", "caltech",
}

_LAB_TITLE_KEYWORDS = {
    "postdoc", "post-doc", "postdoctoral",
    "researcher", "research scientist", "research engineer", "research fellow",
    "principal investigator", "professor", "lecturer",
    "phd", "doctoral", "chargé de recherche", "directeur de recherche",
}

_LAB_DESCRIPTION_KEYWORDS = {
    "publications", "peer-reviewed", "tenure", "phd required",
    "phd preferred", "research group", "principal investigator",
    "scientific community", "academic environment",
}

_STARTUP_KEYWORDS = {
    "startup", "start-up", "early-stage", "early stage",
    "seed funding", "pre-seed", "series a", "series b", "series c",
    "fast-paced", "fast paced", "small team", "growing team",
    "we're building", "we are building", "founding team",
    "co-founder", "cofounder", "disrupting", "venture-backed",
    "venture backed", "backed by", "yc ", "y combinator",
    "techstars", "incubator", "accelerator",
}


def classify_org_type(
    company: str | None,
    title: str | None,
    description: str | None,
) -> str:
    """Classify a job's organization as 'lab', 'startup', or 'corporate'.

    Priority: lab > startup > corporate (default).
    """
    company_lower = (company or "").lower()
    title_lower = (title or "").lower()
    desc_lower = (description or "").lower()

    # --- Lab check (strongest signal: company name, then title, then description) ---
    for kw in _LAB_COMPANY_KEYWORDS:
        if kw in company_lower:
            return "lab"

    for kw in _LAB_TITLE_KEYWORDS:
        if kw in title_lower:
            return "lab"

    # Description alone is weaker — require 2+ hits to avoid false positives
    lab_desc_hits = sum(1 for kw in _LAB_DESCRIPTION_KEYWORDS if kw in desc_lower)
    if lab_desc_hits >= 2:
        return "lab"

    # --- Startup check (company name or description) ---
    for kw in _STARTUP_KEYWORDS:
        if kw in company_lower or kw in desc_lower:
            return "startup"

    return "corporate"



# ---------------------------------------------------------------------------
# Normalization & dedup
# ---------------------------------------------------------------------------

def compute_dedup_hash(title: str, company: str | None) -> str:
    """Normalize title + company and SHA-256 hash for cross-source dedup."""
    normalized = f"{(title or '').strip().lower()}|{(company or '').strip().lower()}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def compute_url_hash(url: str | None) -> str | None:
    """SHA-256 of URL for same-source dedup."""
    if not url:
        return None
    return hashlib.sha256(url.encode()).hexdigest()


def normalize(raw: RawPosting) -> dict[str, Any]:
    """Convert a RawPosting to a dict ready for job insertion."""
    
    return {
        "source": raw.source,
        "source_id": raw.source_id,
        "url": raw.url,
        "url_hash": compute_url_hash(raw.url),
        "title": raw.title,
        "company": raw.company,
        "description": raw.description,
        "location": raw.location,
        "country": raw.country,
        "language": raw.language,
        "seniority": raw.seniority,
        "posted_at": raw.posted_at,
        "raw_data": json.dumps(raw.raw_data) if raw.raw_data else None,
        "dedup_hash": compute_dedup_hash(raw.title, raw.company),
        "org_type": classify_org_type(raw.company, raw.title, raw.description),
    }


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

def _get_adapters(config: dict) -> list[tuple[SourceAdapter, dict]]:
    """Build (adapter, source_config) pairs from config. Lazy imports."""
    from jobscout.sources.wttj import WTTJAdapter
    from jobscout.sources.eures import EURESAdapter
    from jobscout.sources.euraxess import EURAXESSAdapter
    from jobscout.sources.generic_rss import GenericRSSAdapter

    registry: dict[str, type[SourceAdapter]] = {
        "wttj": WTTJAdapter,
        "eures": EURESAdapter,
        "euraxess": EURAXESSAdapter,
        "generic_rss": GenericRSSAdapter,
    }

    pairs = []
    for source_cfg in config.get("sources", []):
        if not source_cfg.get("enabled", True):
            continue
        adapter_key = source_cfg.get("adapter", source_cfg.get("name"))
        adapter_cls = registry.get(adapter_key)
        if adapter_cls is None:
            logger.warning("Unknown adapter %r, skipping", adapter_key)
            continue
        pairs.append((adapter_cls(), source_cfg))
    return pairs


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
    ConnectionError,
    TimeoutError,
)


def _fetch_with_retry(
    adapter: SourceAdapter,
    source_config: dict,
    max_retries: int = 1,
) -> list[RawPosting]:
    """Fetch from an adapter, retrying once on network errors."""
    last_error: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return adapter.fetch(source_config)
        except _NETWORK_ERRORS as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning(
                    "%s: network error (%s), retrying (%d/%d)",
                    adapter.name, exc, attempt + 1, max_retries,
                )
            else:
                raise
    raise last_error  # unreachable, keeps type checker happy


# ---------------------------------------------------------------------------
# Job insertion (includes dedup_hash)
# ---------------------------------------------------------------------------

def _insert_job(conn: sqlite3.Connection, job: dict[str, Any]) -> bool:
    """Insert a single job with dedup_hash and org_type. Returns True if row was inserted."""
    sql = """
        INSERT OR IGNORE INTO jobs
            (source, source_id, url, url_hash, title, company, description,
             location, country, language, seniority, posted_at, raw_data,
             dedup_hash, org_type)
        VALUES
            (:source, :source_id, :url, :url_hash, :title, :company,
             :description, :location, :country, :language, :seniority,
             :posted_at, :raw_data, :dedup_hash, :org_type)
    """
    cur = conn.execute(sql, job)
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_fetch(
    conn: sqlite3.Connection,
    config: dict,
    adapter_names: list[str] | None = None,
) -> int:
    """
    Run a fetch across enabled adapters.

    Creates a runs row, iterates adapters with retry and failure isolation,
    inserts new jobs with cross-source dedup, updates progress.
    Returns the run id (-1 if no adapters enabled).
    """
    pairs = _get_adapters(config)
    if adapter_names:
        pairs = [(a, c) for a, c in pairs if a.name in adapter_names]

    if not pairs:
        logger.warning("No adapters enabled, nothing to fetch")
        return -1

    now = datetime.now(timezone.utc).isoformat()
    source_list = json.dumps([a.name for a, _ in pairs])

    # Create run row
    cur = conn.execute(
        "INSERT INTO runs (status, sources, started_at) VALUES ('running', ?, ?)",
        (source_list, now),
    )
    conn.commit()
    run_id = cur.lastrowid

    total_fetched = 0
    total_new = 0
    errors: list[str] = []
    seen_hashes: set[str] = set()        # cross-adapter in-run dedup

    for i, (adapter, src_cfg) in enumerate(pairs):
        try:
            postings = _fetch_with_retry(adapter, src_cfg)
            logger.info("%s: fetched %d postings", adapter.name, len(postings))

            for raw in postings:
                job = normalize(raw)
                dh = job["dedup_hash"]

                # Cross-source dedup: in-run set + DB check
                if dh in seen_hashes or find_by_dedup_hash(conn, dh):
                    logger.debug("Duplicate skipped: %s @ %s", job["title"], job["company"])
                    continue
                seen_hashes.add(dh)

                if _insert_job(conn, job):
                    total_new += 1
                total_fetched += 1

        except Exception as exc:
            logger.error("%s: failed — %s", adapter.name, exc)
            errors.append(f"{adapter.name}: {exc}")

        # Update progress after each adapter
        progress = (i + 1) / len(pairs)
        conn.execute(
            "UPDATE runs SET progress = ?, total_jobs = ?, new_jobs = ? WHERE id = ?",
            (progress, total_fetched, total_new, run_id),
        )
        conn.commit()

    # Finalize run
    status = "failed" if len(errors) == len(pairs) else "completed"
    conn.execute(
        "UPDATE runs SET status = ?, completed_at = ?, error = ? WHERE id = ?",
        (status, datetime.now(timezone.utc).isoformat(),
         "\n".join(errors) if errors else None, run_id),
    )
    conn.commit()

    logger.info(
        "Run %d done: %d new / %d fetched (%d adapter errors)",
        run_id, total_new, total_fetched, len(errors),
    )
    return run_id