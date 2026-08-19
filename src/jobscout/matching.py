"""Embedding-based matching engine: faceted scoring between profiles and jobs."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from jobscout.db import get_profile, get_all_jobs
from jobscout.embedder import Embedder, to_blob, from_blob
from jobscout.profiles import RuleBasedExtractor, find_required_years

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "skills": 0.60,
    "domain": 0.40,
}

# --- Seniority as years (M7b) ----------------------------------------------
#
# The comparison is deliberately asymmetric. Being short of a requirement is
# disqualifying past a point; being over it is a preference, never a filter.

DEFAULT_GATE_YEARS = 2.0     # gap beyond this filters the job out
DEFAULT_FILTER_ON_INFERRED = False  # a title guess penalises but does not filter
NEAR_MISS_FLOOR = 0.75       # multiplier at exactly one year short
GATE_FLOOR = 0.25            # multiplier at exactly the gate
BEYOND_GATE_DECAY = 0.10     # per year past the gate, for ranking stretch roles
# Raised from 0.05: at 0.05 everything past the gate landed in a band too
# narrow to order. These jobs are hidden by default; the floor only has to
# let them rank sensibly against each other once stretch roles are shown.
BEYOND_GATE_MIN = 0.15

OVER_GRACE_YEARS = 2.0       # years of overqualification that cost nothing
OVER_DECAY_PER_YEAR = 0.02   # and a token penalty after that
OVER_FLOOR = 0.85

# Where a requirement came from, best evidence first.
#   api         - the source's own structured field
#   description - stated in the posting's prose (M7c)
#   title       - guessed from title wording
#   none        - nothing to go on
# The first two are stated requirements, so either may remove a job from
# view. Prose is evidence, not a guess: a posting that writes "au moins 3
# ans d'experience" has stated a floor as plainly as an API field does.
STATED_SOURCES: frozenset[str] = frozenset({"api", "description"})

# Fallback when a source states no requirement: read the title. Coarse by
# design — two buckets, not a ladder, because that is all a title supports.
TITLE_YEARS_ENTRY = 0.0
TITLE_YEARS_SENIOR = 5.0

# Checked before the senior patterns: "Stagiaire Data Architect" is an
# internship, not an architect role.
ENTRY_TITLE_PATTERN = re.compile(
    r"\b("
    r"junior|jr|entry[\s-]?level|graduate|new\s?grad|trainee|"          # EN
    r"intern|internship|apprentice|apprenticeship|student|"
    r"stagiaire|stage|alternance|alternant|apprenti|débutant|debutant|"  # FR
    r"jeune\s+diplômé|jeune\s+diplome|"
    r"praktikum|praktikant|werkstudent|auszubildende|einsteiger|berufseinsteiger"  # DE
    r")\b",
    re.IGNORECASE,
)

SENIOR_TITLE_PATTERN = re.compile(
    r"\b("
    r"senior|sr|lead|principal|staff|architect|expert|head\s+of|"        # EN
    r"confirmé|confirme|expérimenté|experimente|responsable|"            # FR
    r"chef\s+de\s+projet|chef\s+d'équipe|architecte|"
    r"leitender|leiter|teamleiter|architekt|erfahren"                    # DE
    r")\b",
    re.IGNORECASE,
)

# Map job posting language codes to profile spoken language names
LANG_CODE_TO_NAME: dict[str, str] = {
    "en": "English", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ar": "Arabic",
}

PROFILE_JSON_FIELDS: set[str] = {
    "skills", "domains", "languages", "target_locations",
    "company_types", "position_types",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def build_facet_text(items: list[str]) -> str:
    """Turn a list of skills or domains into embeddable text."""
    return ", ".join(items) if items else ""


@dataclass
class SeniorityVerdict:
    """The seniority facet's full reasoning, so a score can be audited."""

    multiplier: float
    filtered: bool
    required_years: float | None
    candidate_years: float | None
    gap: float | None
    source: str              # 'api' | 'description' | 'title' | 'none'
    # The description text that produced the number, when source is
    # 'description'. Shown in the UI so a bad regex match is visible rather
    # than silently scoring a job.
    snippet: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "filtered": self.filtered,
            "required_years": self.required_years,
            "candidate_years": self.candidate_years,
            "gap": self.gap,
            "source": self.source,
            "snippet": self.snippet,
        }


def infer_years_from_title(title: str | None) -> float | None:
    """Guess a requirement from title wording. None when the title says nothing."""
    if not title:
        return None
    if ENTRY_TITLE_PATTERN.search(title):
        return TITLE_YEARS_ENTRY
    if SENIOR_TITLE_PATTERN.search(title):
        return TITLE_YEARS_SENIOR
    return None


def score_seniority_years(
    required_years: float | None,
    candidate_years: float | None,
    gate: float = DEFAULT_GATE_YEARS,
) -> tuple[float, bool]:
    """Compare years. Returns (multiplier, filtered).

    Underqualification decays steeply and eventually disqualifies;
    overqualification barely registers and never filters.
    """
    if required_years is None or candidate_years is None:
        return 1.0, False           # nothing to compare — stay neutral

    gap = required_years - candidate_years

    if gap <= 0:
        excess = -gap
        if excess <= OVER_GRACE_YEARS:
            return 1.0, False
        decayed = 1.0 - OVER_DECAY_PER_YEAR * (excess - OVER_GRACE_YEARS)
        return max(OVER_FLOOR, round(decayed, 4)), False

    if gap > gate:
        # Still scored, so stretch roles rank sensibly among themselves.
        decayed = GATE_FLOOR - BEYOND_GATE_DECAY * (gap - gate)
        return max(BEYOND_GATE_MIN, round(decayed, 4)), True

    if gap <= 1.0:
        # Near miss: mild.
        return round(1.0 - (1.0 - NEAR_MISS_FLOOR) * gap, 4), False

    # Between a year short and the gate: steep.
    span = max(gate - 1.0, 1e-9)
    fraction = (gap - 1.0) / span
    return round(NEAR_MISS_FLOOR - (NEAR_MISS_FLOOR - GATE_FLOOR) * fraction, 4), False


def resolve_seniority(
    job: Any,
    candidate_years: float | None,
    gate: float = DEFAULT_GATE_YEARS,
    filter_on_inferred: bool = DEFAULT_FILTER_ON_INFERRED,
) -> SeniorityVerdict:
    """Work out the requirement, then score it.

    Falls through in order: the source's structured field, then the posting's
    own prose, then the title, then nothing. A job we know nothing about is
    never filtered on that ignorance.
    """
    required = _row_get(job, "required_years_min")
    source = "api"
    snippet = None

    if required is not None and float(required) == 0.0:
        # WTTJ sends experience_level_minimum: 0 for postings that state no
        # requirement at all, so the field carries "unknown" and "genuinely
        # zero" in the same value. Taken literally it stops the chain at `api`,
        # the posting's own prose is never read, and a job demanding "5+ years"
        # is scored and labelled as stating zero. Treat it as unset instead:
        # the description may say what the field does not.
        logger.info(
            "seniority: api required_years_min is 0 for %r — the source uses 0 "
            "for 'not stated', so treating it as unset and reading the "
            "description instead",
            _row_get(job, "title"),
        )
        required = None

    if required is None:
        # The structured field is null far more often than postings are
        # actually silent about experience, so read what the posting says.
        found = find_required_years(
            _row_get(job, "description"), _row_get(job, "language"),
        )
        if found is not None:
            required, source, snippet = found.years, "description", found.snippet

    if required is None:
        required = infer_years_from_title(_row_get(job, "title"))
        source = "title" if required is not None else "none"

    if required is None:
        return SeniorityVerdict(1.0, False, None, candidate_years, None, "none")

    required = float(required)
    multiplier, filtered = score_seniority_years(required, candidate_years, gate)

    # Only a stated requirement may remove a job from view. A title guess keeps
    # its penalty but stays visible: a wrong filter loses the job, a wrong
    # penalty only costs rank position.
    if filtered and source not in STATED_SOURCES and not filter_on_inferred:
        filtered = False

    gap = None if candidate_years is None else round(required - candidate_years, 4)
    return SeniorityVerdict(
        multiplier, filtered, required, candidate_years, gap, source, snippet,
    )


def _row_get(row: Any, key: str) -> Any:
    """Read a key from a sqlite3.Row or a plain dict, tolerating absence."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of L2-normalized vectors (= cosine similarity)."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Pre-filtering
# ---------------------------------------------------------------------------

def passes_filters(job: sqlite3.Row, profile: dict) -> bool:
    """Return True if the job passes all profile filters."""
    # Location: substring match — "Paris" matches "Paris, France"
    target_locations = profile.get("target_locations", [])
    if target_locations:
        job_loc = (job["location"] or "").lower()
        job_country = (job["country"] or "").lower()
        if not any(
            loc.lower() in job_loc or loc.lower() in job_country
            for loc in target_locations
        ):
            return False

    # Language: job posting language ∈ profile's spoken languages
    spoken = profile.get("languages", [])
    if spoken:
        job_lang = (job["language"] or "").lower()
        job_lang_name = LANG_CODE_TO_NAME.get(job_lang, "")
        if job_lang_name and job_lang_name not in spoken:
            return False

    return True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_job(
    profile_skills_emb: np.ndarray,
    profile_domain_emb: np.ndarray,
    job_skills_emb: np.ndarray,
    job_domain_emb: np.ndarray,
    profile_skills: list[str],
    job_skills: list[str],
    weights: dict[str, float] | None = None,
    seniority: SeniorityVerdict | None = None,
) -> dict[str, Any]:
    """Score a single job against a profile. Returns the explainability payload."""
    w = weights or DEFAULT_WEIGHTS
    if seniority is None:
        # Nothing known about the requirement — neutral, and never filtered.
        seniority = SeniorityVerdict(1.0, False, None, None, None, "none")

    skills_score = max(0.0, cosine_similarity(profile_skills_emb, job_skills_emb))
    domain_score = max(0.0, cosine_similarity(profile_domain_emb, job_domain_emb))

    base_score = w["skills"] * skills_score + w["domain"] * domain_score
    final = base_score * seniority.multiplier

    matched_skills = sorted(set(profile_skills) & set(job_skills))

    return {
        "skills_score": round(skills_score, 4),
        "domain_score": round(domain_score, 4),
        "seniority_multiplier": round(seniority.multiplier, 4),
        # Six places, not four: at four, the seniority multiplier packed 107
        # distinct scores into a 0.0057-wide window and the ranking stopped
        # ranking. The extra places cost nothing and break the ties.
        "final_score": round(final, 6),
        "weights": w,
        "matched_skills": matched_skills,
        "seniority": seniority,
        "filtered": seniority.filtered,
    }


# ---------------------------------------------------------------------------
# DB helpers (M3-specific)
# ---------------------------------------------------------------------------

def _deserialize_profile(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a profile row to a dict with parsed JSON fields."""
    d = dict(row)
    for key in PROFILE_JSON_FIELDS:
        val = d.get(key)
        if isinstance(val, str):
            d[key] = json.loads(val)
    return d


def _extract_job_facets(
    conn: sqlite3.Connection,
    job: sqlite3.Row,
    extractor: RuleBasedExtractor,
) -> dict[str, Any]:
    """Extract skills/domains from a job's title + description, cache in DB."""
    job_dict = dict(job)
    text = (job_dict.get("title") or "") + "\n" + (job_dict.get("description") or "")
    extracted = extractor.extract_from_text(text)

    # Cache extraction results
    conn.execute(
        "UPDATE jobs SET skills = ?, domains = ? WHERE id = ?",
        (json.dumps(extracted["skills"]), json.dumps(extracted["domains"]), job_dict["id"]),
    )
    return extracted


def _embed_job_facets(
    conn: sqlite3.Connection,
    job: sqlite3.Row,
    skills_text: str,
    domain_text: str,
    embedder: Embedder,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed job facets and cache in DB. Returns (skills_emb, domain_emb)."""
    # Check cache — BLOBs are NULL until first embedding
    if job["skills_embedding"] is not None and job["domain_embedding"] is not None:
        return (
            from_blob(job["skills_embedding"]),
            from_blob(job["domain_embedding"]),
        )

    # Embed as passages (jobs are documents, not queries)
    skills_emb = embedder.embed(skills_text or "general", is_query=False)
    domain_emb = embedder.embed(domain_text or "general", is_query=False)

    # Cache
    conn.execute(
        "UPDATE jobs SET skills_embedding = ?, domain_embedding = ? WHERE id = ?",
        (to_blob(skills_emb), to_blob(domain_emb), job["id"]),
    )
    return skills_emb, domain_emb


def _upsert_match(
    conn: sqlite3.Connection,
    profile_id: int,
    job_id: int,
    result: dict[str, Any],
) -> None:
    """Insert or update a match result."""
    conn.execute(
        """
        INSERT INTO matches (profile_id, job_id, score, facet_scores, explanation)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, job_id) DO UPDATE SET
            score = excluded.score,
            facet_scores = excluded.facet_scores,
            explanation = excluded.explanation
        """,
        (
            profile_id,
            job_id,
            result["final_score"],
            json.dumps({
                "skills": result["skills_score"],
                "domain": result["domain_score"],
                "seniority": result["seniority_multiplier"],
                # Underscore-prefixed: detail, not a 0-1 facet score. Consumers
                # that render facet bars skip these.
                "_seniority": result["seniority"].as_dict(),
            }),
            json.dumps(result["matched_skills"]),
        ),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_matching(
    conn: sqlite3.Connection,
    profile_name: str,
    embedder: Embedder | None = None,
    weights: dict[str, float] | None = None,
    org_types: list[str] | None = None,
    candidate_years: float | None = None,
    gate_years: float = DEFAULT_GATE_YEARS,
    filter_on_inferred: bool = DEFAULT_FILTER_ON_INFERRED,
) -> list[dict[str, Any]]:
    """Run the full matching pipeline for a profile against all jobs.

    Steps:
        1. Load profile, deserialize JSON fields
        2. Embed profile facets as queries
        3. Cache profile embeddings in DB
        4. Load all jobs, apply pre-filters
        5. Per job: extract facets → embed as passages → score
        6. Rank by final_score descending
        7. Store match results

    Returns a ranked list of match results (highest score first).
    """
    # 1. Load and deserialize profile
    profile_row = get_profile(conn, profile_name)
    if not profile_row:
        raise ValueError(f"Profile not found: {profile_name}")
    profile = _deserialize_profile(profile_row)

    # 2. Embed profile facets (profiles are queries — what you're looking for)
    if embedder is None:
        embedder = Embedder()

    profile_skills_text = build_facet_text(profile["skills"])
    profile_domain_text = build_facet_text(profile["domains"])

    profile_skills_emb = embedder.embed(profile_skills_text or "general", is_query=True)
    profile_domain_emb = embedder.embed(profile_domain_text or "general", is_query=True)

    # 3. Cache profile embeddings
    conn.execute(
        "UPDATE profiles SET skills_embedding = ?, domain_embedding = ? WHERE id = ?",
        (to_blob(profile_skills_emb), to_blob(profile_domain_emb), profile["id"]),
    )
    conn.commit()

    # 4. Load and filter jobs
    all_jobs = get_all_jobs(conn)
    jobs = [j for j in all_jobs if passes_filters(j, profile)]

    if org_types:
        jobs = [j for j in jobs if (j["org_type"] or "corporate") in org_types]

    # 5. Extract → embed → score each job
    extractor = RuleBasedExtractor()
    results = []
    has_source_column = "seniority_source" in {
        row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
    }

    for job in jobs:
        # Extract facets from title + description
        job_extracted = _extract_job_facets(conn, job, extractor)

        # Build embeddable text
        job_skills_text = build_facet_text(job_extracted["skills"])
        job_domain_text = build_facet_text(job_extracted["domains"])

        # Embed (uses cache if available)
        job_skills_emb, job_domain_emb = _embed_job_facets(
            conn, job, job_skills_text, job_domain_text, embedder,
        )

        # Seniority: years arithmetic, api -> description -> title -> nothing
        verdict = resolve_seniority(
            job, candidate_years, gate_years, filter_on_inferred,
        )
        if has_source_column and _row_get(job, "seniority_source") != verdict.source:
            conn.execute(
                "UPDATE jobs SET seniority_source = ? WHERE id = ?",
                (verdict.source, job["id"]),
            )

        # Score
        result = score_job(
            profile_skills_emb=profile_skills_emb,
            profile_domain_emb=profile_domain_emb,
            job_skills_emb=job_skills_emb,
            job_domain_emb=job_domain_emb,
            profile_skills=profile["skills"],
            job_skills=job_extracted["skills"],
            weights=weights,
            seniority=verdict,
        )
        result["job_id"] = job["id"]
        result["job_title"] = job["title"]
        result["job_company"] = job["company"]
        results.append(result)

    # Batch commit extractions + embeddings
    conn.commit()

    # 6. Rank by score
    results.sort(key=lambda r: r["final_score"], reverse=True)

    # 7. Store matches. Filtered ("stretch") jobs are stored too — hiding them
    # by default is a view concern, and the user can always ask to see them.
    for result in results:
        _upsert_match(conn, profile["id"], result["job_id"], result)
    conn.commit()

    filtered = sum(1 for r in results if r["filtered"])
    if filtered:
        logger.info(
            "seniority gate (%.1fy, candidate %s): %d of %d jobs are stretch roles",
            gate_years,
            "unset" if candidate_years is None else f"{candidate_years:g}y",
            filtered, len(results),
        )

    return results