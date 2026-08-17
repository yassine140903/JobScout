"""Embedding-based matching engine: faceted scoring between profiles and jobs."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np

from jobscout.db import get_profile, get_all_jobs
from jobscout.embedder import Embedder, to_blob, from_blob
from jobscout.profiles import RuleBasedExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENIORITY_ORDER: dict[str, int] = {
    "intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "principal": 5,
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "skills": 0.60,
    "domain": 0.40,
}

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


def score_seniority(profile_level: str, job_level: str) -> float:
    """Seniority multiplier. 1.0 = exact match, decays sharply with distance."""
    p = SENIORITY_ORDER.get(profile_level, 1)
    j = SENIORITY_ORDER.get(job_level, 1)
    distance = abs(p - j)
    return {0: 1.0, 1: 0.85, 2: 0.6, 3: 0.4}.get(distance, 0.3)


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
    profile_seniority: str,
    job_seniority: str,
    profile_skills: list[str],
    job_skills: list[str],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a single job against a profile. Returns the explainability payload."""
    w = weights or DEFAULT_WEIGHTS

    skills_score = max(0.0, cosine_similarity(profile_skills_emb, job_skills_emb))
    domain_score = max(0.0, cosine_similarity(profile_domain_emb, job_domain_emb))
    seniority_multiplier = score_seniority(profile_seniority, job_seniority)

    base_score = w["skills"] * skills_score + w["domain"] * domain_score
    final = base_score * seniority_multiplier

    matched_skills = sorted(set(profile_skills) & set(job_skills))

    return {
        "skills_score": round(skills_score, 4),
        "domain_score": round(domain_score, 4),
        "seniority_multiplier": round(seniority_multiplier, 4),
        "final_score": round(final, 4),
        "weights": w,
        "matched_skills": matched_skills,
        "profile_seniority": profile_seniority,
        "job_seniority": job_seniority,
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

    # Prefer the job's declared seniority over extracted
    if job_dict.get("seniority"):
        extracted["seniority"] = job_dict["seniority"]

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

        # Score
        result = score_job(
            profile_skills_emb=profile_skills_emb,
            profile_domain_emb=profile_domain_emb,
            job_skills_emb=job_skills_emb,
            job_domain_emb=job_domain_emb,
            profile_seniority=profile.get("seniority") or "junior",
            job_seniority=job_extracted.get("seniority") or "junior",
            profile_skills=profile["skills"],
            job_skills=job_extracted["skills"],
            weights=weights,
        )
        result["job_id"] = job["id"]
        result["job_title"] = job["title"]
        result["job_company"] = job["company"]
        results.append(result)

    # Batch commit extractions + embeddings
    conn.commit()

    # 6. Rank by score
    results.sort(key=lambda r: r["final_score"], reverse=True)

    # 7. Store matches
    for result in results:
        _upsert_match(conn, profile["id"], result["job_id"], result)
    conn.commit()

    return results