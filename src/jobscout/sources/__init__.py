"""Source adapters: ABC, normalization, dedup, and fetch orchestrator."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from jobscout.db import find_by_dedup_hash
from jobscout.textclean import clean_description

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
    # M7b: structured requirements. Adapters that cannot supply these leave them
    # None — a missing requirement is not a requirement of zero.
    required_years_min: float | None = None   # minimum years of experience
    education_level: str | None = None        # source enum, e.g. "BAC_5"
    salary_yearly_min: int | None = None
    salary_currency: str | None = None
    seniority_source: str | None = None       # 'api' when the source stated it
    # M7d: which text the description came from, and whether the posting is
    # still reachable. Adapters with a single description path leave both None.
    description_source: str | None = None     # 'detail' | 'blurb' | 'none'
    delisted_at: str | None = None            # set when the posting 404s


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
# Position-type classification
# ---------------------------------------------------------------------------

# Matched on word boundaries, not as bare substrings. As substrings, "intern"
# matched "internal" and "international", and "mission" matched ordinary prose
# ("notre mission est de..."), which classified two senior engineering roles as
# internships. Inflections that matter are spelled out rather than left to a
# prefix match, because a prefix match is what caused the bug.
_INTERNSHIP_KEYWORDS = {
    # EN
    "intern", "interns", "internship", "internships",
    "apprentice", "apprentices", "apprenticeship", "apprenticeships",
    "trainee", "trainees", "working student", "placement student",
    # FR
    "stage", "stages", "stagiaire", "stagiaires",
    "alternance", "alternant", "alternante", "alternants", "alternantes",
    "apprenti", "apprentie", "apprentis", "apprenties",
    # DE
    "praktikum", "praktika", "praktikant", "praktikantin",
    "werkstudent", "werkstudentin", "auszubildende", "ausbildung",
}

_FREELANCE_KEYWORDS = {
    # EN
    "freelance", "freelancer", "freelancers", "free-lance",
    "contractor", "contractors", "self-employed", "contract position",
    "independent contractor", "1099",
    # FR. Bare "mission" is deliberately absent: every consultancy posting says
    # "notre mission", and it classified staff jobs as freelance.
    "freelances", "consultant indépendant", "consultante indépendante",
    "travailleur indépendant", "auto-entrepreneur", "portage salarial",
    # DE
    "freiberuflich", "freiberufler", "selbständig", "selbstständig",
}


def _keyword_pattern(keywords: set[str]) -> re.Pattern[str]:
    """One alternation over a keyword set, word-bounded and hyphen-tolerant."""
    alternatives = sorted(
        (r"[\s\-]+".join(re.escape(part) for part in kw.split()) for kw in keywords),
        key=len,
        reverse=True,          # longest first, so "internship" wins over "intern"
    )
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


_INTERNSHIP_RE = _keyword_pattern(_INTERNSHIP_KEYWORDS)
_FREELANCE_RE = _keyword_pattern(_FREELANCE_KEYWORDS)


def classify_position_type(
    title: str | None,
    description: str | None,
) -> str:
    """Classify a job as 'internship', 'freelance', or 'job' (default).

    Title markers are checked before description markers, so a title saying
    "Freelance" is not overridden by the word "stage" further down the body.
    A marker that appears only in the description still classifies, as before -
    this pass changed how the keywords match, not where they are looked for.
    """
    title_text = title or ""
    description_text = description or ""

    if _INTERNSHIP_RE.search(title_text):
        return "internship"
    if _FREELANCE_RE.search(title_text):
        return "freelance"
    if _INTERNSHIP_RE.search(description_text):
        return "internship"
    if _FREELANCE_RE.search(description_text):
        return "freelance"
    return "job"


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
    """Convert a RawPosting to a dict ready for job insertion.

    Descriptions are cleaned here so every posting is stored as structured
    plain text regardless of which adapter produced it.
    """
    description = clean_description(raw.description) or None

    # Classify on the cleaned text: keyword checks should not have to see
    # through style attributes and entity escapes.
    return {
        "source": raw.source,
        "source_id": raw.source_id,
        "url": raw.url,
        "url_hash": compute_url_hash(raw.url),
        "title": raw.title,
        "company": raw.company,
        "description": description,
        "location": raw.location,
        "country": raw.country,
        "language": raw.language,
        "seniority": raw.seniority,
        "posted_at": raw.posted_at,
        "raw_data": json.dumps(raw.raw_data) if raw.raw_data else None,
        "dedup_hash": compute_dedup_hash(raw.title, raw.company),
        "org_type": classify_org_type(raw.company, raw.title, description),
        "position_type": classify_position_type(raw.title, description),
        "required_years_min": raw.required_years_min,
        "education_level": raw.education_level,
        "salary_yearly_min": raw.salary_yearly_min,
        "salary_currency": raw.salary_currency,
        "seniority_source": raw.seniority_source,
        "description_source": raw.description_source,
        "delisted_at": raw.delisted_at,
    }


# ---------------------------------------------------------------------------
# Profile-driven config enrichment
# ---------------------------------------------------------------------------

def enrich_config_from_profile(config: dict, profile_row) -> dict:
    """Inject profile-derived keywords into source configs.

    For any source entry with an empty ``keywords`` list, fills it with
    the profile's domains + top skills (capped at 8 terms).  EURAXESS uses
    ``keyword`` (singular) — handled separately.

    Locations are deliberately not injected: each API expects its own
    format (EURES wants NUTS codes, WTTJ something else), so location
    filtering is left to ``passes_filters()`` during matching.

    Returns a deep-copied config so the original is not mutated.
    """
    config = copy.deepcopy(config)

    # Parse profile fields (may be JSON strings or already lists)
    def _parse(val):
        if isinstance(val, str):
            return json.loads(val or "[]")
        return val or []

    skills = _parse(profile_row["skills"])
    domains = _parse(profile_row["domains"])

    # Domains first (broader), then skills — deduplicated, capped
    keywords = list(dict.fromkeys(domains + skills))[:8]

    for source in config.get("sources", []):
        # keywords (list) — WTTJ, EURES
        if "keywords" in source and not source["keywords"]:
            source["keywords"] = keywords
        # keyword (singular) — EURAXESS
        if "keyword" in source and not source["keyword"]:
            source["keyword"] = " ".join(keywords[:3]) if keywords else None

    return config


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
    """Insert a single job with dedup_hash, org_type, and position_type.

    Returns True if row was inserted.
    """
    sql = """
        INSERT OR IGNORE INTO jobs
            (source, source_id, url, url_hash, title, company, description,
             location, country, language, seniority, posted_at, raw_data,
             dedup_hash, org_type, position_type,
             required_years_min, education_level, salary_yearly_min,
             salary_currency, seniority_source, description_source, delisted_at)
        VALUES
            (:source, :source_id, :url, :url_hash, :title, :company,
             :description, :location, :country, :language, :seniority,
             :posted_at, :raw_data, :dedup_hash, :org_type, :position_type,
             :required_years_min, :education_level, :salary_yearly_min,
             :salary_currency, :seniority_source, :description_source,
             :delisted_at)
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