"""Tests for M3: embedding + matching engine."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from jobscout.db import init_db, migrate_m2, migrate_m3, insert_jobs_bulk, upsert_profile
from jobscout.embedder import to_blob, from_blob
from jobscout.matching import (
    build_facet_text,
    score_seniority,
    cosine_similarity,
    passes_filters,
    score_job,
    run_matching,
    SENIORITY_ORDER,
)
from jobscout.profiles import RuleBasedExtractor


# ---------------------------------------------------------------------------
# Fake embedder for deterministic tests
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Returns deterministic normalized vectors. Same text → same vector."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.call_count = 0

    def embed(self, text: str, is_query: bool = False) -> np.ndarray:
        self.call_count += 1
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return np.array([self.embed(t, is_query) for t in texts])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DB with all migrations applied."""
    c = init_db(":memory:")
    migrate_m2(c)
    migrate_m3(c)
    return c


@pytest.fixture
def sample_profile(conn):
    """A mid-level ML engineer profile."""
    profile = {
        "name": "ml-engineer",
        "raw_text": "Experienced ML engineer with Python and PyTorch.",
        "skills": ["python", "pytorch", "docker", "kubernetes", "sql"],
        "domains": ["machine learning", "mlops"],
        "seniority": "mid",
        "languages": ["English", "French"],
        "target_locations": ["Paris", "Berlin"],
        "company_types": [],
        "position_types": ["job"],
    }
    upsert_profile(conn, profile)
    return profile


@pytest.fixture
def sample_jobs(conn):
    """Mix of jobs: some should match well, some poorly, some filtered out."""
    jobs = [
        {
            "source": "fixture", "source_id": "j1",
            "url": "https://example.com/j1", "url_hash": "h1",
            "title": "Senior ML Engineer",
            "company": "AI Corp",
            "description": "We need a Python and PyTorch expert for MLOps pipelines. "
                           "Experience with Docker and Kubernetes required.",
            "location": "Paris, France", "country": "FR",
            "language": "en", "seniority": "senior",
            "posted_at": "2026-01-01", "raw_data": None,
        },
        {
            "source": "fixture", "source_id": "j2",
            "url": "https://example.com/j2", "url_hash": "h2",
            "title": "Marketing Manager",
            "company": "AdTech Inc",
            "description": "Lead our digital marketing campaigns. SEO, SEM, "
                           "and social media expertise required.",
            "location": "Berlin, Germany", "country": "DE",
            "language": "en", "seniority": "senior",
            "posted_at": "2026-01-02", "raw_data": None,
        },
        {
            "source": "fixture", "source_id": "j3",
            "url": "https://example.com/j3", "url_hash": "h3",
            "title": "Data Scientist",
            "company": "DataCo",
            "description": "Python, SQL, and machine learning skills needed. "
                           "Work on NLP and computer vision projects.",
            "location": "Paris, France", "country": "FR",
            "language": "fr", "seniority": "mid",
            "posted_at": "2026-01-03", "raw_data": None,
        },
        {
            "source": "fixture", "source_id": "j4",
            "url": "https://example.com/j4", "url_hash": "h4",
            "title": "Frontend Developer",
            "company": "WebShop",
            "description": "React and TypeScript developer needed for e-commerce platform.",
            "location": "Tokyo, Japan", "country": "JP",
            "language": "en", "seniority": "junior",
            "posted_at": "2026-01-04", "raw_data": None,
        },
        {
            "source": "fixture", "source_id": "j5",
            "url": "https://example.com/j5", "url_hash": "h5",
            "title": "Ingénieur Machine Learning",
            "company": "FrenchAI",
            "description": "Développer des modèles de deep learning avec Python et PyTorch. "
                           "Expérience en MLOps souhaitée.",
            "location": "Paris, France", "country": "FR",
            "language": "fr", "seniority": "mid",
            "posted_at": "2026-01-05", "raw_data": None,
        },
    ]
    insert_jobs_bulk(conn, jobs)
    return jobs


# ---------------------------------------------------------------------------
# Test: schema migration
# ---------------------------------------------------------------------------

class TestMigration:
    def test_m3_adds_profile_embedding_columns(self, conn):
        cols = {row[1] for row in conn.execute("PRAGMA table_info(profiles)").fetchall()}
        assert "skills_embedding" in cols
        assert "domain_embedding" in cols

    def test_m3_adds_job_columns(self, conn):
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "skills" in cols
        assert "domains" in cols
        assert "skills_embedding" in cols
        assert "domain_embedding" in cols

    def test_m3_idempotent(self, conn):
        # Should not raise on second call
        migrate_m3(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "skills_embedding" in cols


# ---------------------------------------------------------------------------
# Test: blob serialization
# ---------------------------------------------------------------------------

class TestBlobSerialization:
    def test_roundtrip(self):
        original = np.random.randn(768).astype(np.float32)
        blob = to_blob(original)
        restored = from_blob(blob)
        np.testing.assert_array_almost_equal(original, restored)

    def test_blob_is_bytes(self):
        vec = np.ones(768, dtype=np.float32)
        blob = to_blob(vec)
        assert isinstance(blob, bytes)
        assert len(blob) == 768 * 4  # float32 = 4 bytes


# ---------------------------------------------------------------------------
# Test: utilities
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_build_facet_text(self):
        assert build_facet_text(["python", "docker"]) == "python, docker"
        assert build_facet_text([]) == ""

    def test_cosine_similarity_identical(self):
        vec = np.random.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        a = np.zeros(768, dtype=np.float32)
        b = np.zeros(768, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test: seniority scoring
# ---------------------------------------------------------------------------

class TestSeniorityScoring:
    def test_exact_match(self):
        assert score_seniority("mid", "mid") == 1.0

    def test_one_step(self):
        assert score_seniority("mid", "senior") == 0.6
        assert score_seniority("senior", "mid") == 0.6

    def test_two_steps(self):
        assert score_seniority("junior", "senior") == 0.3

    def test_three_plus_steps(self):
        assert score_seniority("intern", "lead") == 0.1
        assert score_seniority("junior", "principal") == 0.1

    def test_unknown_defaults_to_junior(self):
        # Unknown maps to index 1 (junior)
        assert score_seniority("unknown", "junior") == 1.0


# ---------------------------------------------------------------------------
# Test: pre-filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def _make_job(self, location="Paris, France", country="FR", language="en"):
        return sqlite3.Row  # We need a dict-like; use a real row from DB

    def test_passes_location_match(self, conn, sample_jobs):
        profile = {"target_locations": ["Paris"], "languages": ["English"]}
        jobs = conn.execute("SELECT * FROM jobs WHERE source_id = 'j1'").fetchall()
        assert passes_filters(jobs[0], profile) is True

    def test_fails_location(self, conn, sample_jobs):
        profile = {"target_locations": ["Paris"], "languages": ["English"]}
        jobs = conn.execute("SELECT * FROM jobs WHERE source_id = 'j4'").fetchall()
        assert passes_filters(jobs[0], profile) is False  # Tokyo

    def test_fails_language(self, conn, sample_jobs):
        profile = {"target_locations": ["Paris"], "languages": ["German"]}
        jobs = conn.execute("SELECT * FROM jobs WHERE source_id = 'j1'").fetchall()
        assert passes_filters(jobs[0], profile) is False  # English posting, German speaker only

    def test_empty_locations_passes_all(self, conn, sample_jobs):
        profile = {"target_locations": [], "languages": []}
        jobs = conn.execute("SELECT * FROM jobs WHERE source_id = 'j4'").fetchall()
        assert passes_filters(jobs[0], profile) is True

    def test_country_code_match(self, conn, sample_jobs):
        profile = {"target_locations": ["FR"], "languages": []}
        jobs = conn.execute("SELECT * FROM jobs WHERE source_id = 'j1'").fetchall()
        assert passes_filters(jobs[0], profile) is True


# ---------------------------------------------------------------------------
# Test: job facet extraction
# ---------------------------------------------------------------------------

class TestJobExtraction:
    def test_extract_from_text_skills(self):
        extractor = RuleBasedExtractor()
        text = "Senior ML Engineer\nPython and PyTorch required. Docker experience."
        result = extractor.extract_from_text(text)
        assert "python" in result["skills"]
        assert "pytorch" in result["skills"]
        assert "docker" in result["skills"]

    def test_extract_from_text_domains(self):
        extractor = RuleBasedExtractor()
        text = "Machine learning and MLOps position"
        result = extractor.extract_from_text(text)
        assert "machine learning" in result["domains"]
        assert "mlops" in result["domains"]

    def test_extract_from_text_seniority(self):
        extractor = RuleBasedExtractor()
        text = "Senior Software Engineer needed"
        result = extractor.extract_from_text(text)
        assert result["seniority"] == "senior"

    def test_no_languages_in_extract_from_text(self):
        extractor = RuleBasedExtractor()
        result = extractor.extract_from_text("Python developer in Paris")
        assert "languages" not in result


# ---------------------------------------------------------------------------
# Test: score_job
# ---------------------------------------------------------------------------

class TestScoreJob:
    def test_identical_vectors_max_score(self):
        vec = np.random.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        result = score_job(
            profile_skills_emb=vec, profile_domain_emb=vec,
            job_skills_emb=vec, job_domain_emb=vec,
            profile_seniority="mid", job_seniority="mid",
            profile_skills=["python"], job_skills=["python"],
        )
        assert result["final_score"] == pytest.approx(1.0, abs=1e-3)
        assert result["matched_skills"] == ["python"]

    def test_custom_weights(self):
        vec = np.random.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        weights = {"skills": 1.0, "domain": 0.0, "seniority": 0.0}
        result = score_job(
            profile_skills_emb=vec, profile_domain_emb=vec,
            job_skills_emb=vec, job_domain_emb=vec,
            profile_seniority="junior", job_seniority="principal",
            profile_skills=[], job_skills=[],
            weights=weights,
        )
        # Seniority mismatch doesn't matter with weight 0
        assert result["final_score"] == pytest.approx(1.0, abs=1e-3)

    def test_no_skill_overlap(self):
        vec = np.random.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        result = score_job(
            profile_skills_emb=vec, profile_domain_emb=vec,
            job_skills_emb=vec, job_domain_emb=vec,
            profile_seniority="mid", job_seniority="mid",
            profile_skills=["python", "pytorch"],
            job_skills=["react", "typescript"],
        )
        assert result["matched_skills"] == []


# ---------------------------------------------------------------------------
# Test: run_matching (end-to-end with FakeEmbedder)
# ---------------------------------------------------------------------------

class TestRunMatching:
    def test_returns_ranked_results(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        results = run_matching(conn, "ml-engineer", embedder=embedder)
        assert len(results) > 0
        # Results should be sorted descending
        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_filtered_jobs_excluded(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        results = run_matching(conn, "ml-engineer", embedder=embedder)
        job_titles = [r["job_title"] for r in results]
        # Tokyo job should be filtered out (not in Paris/Berlin)
        assert "Frontend Developer" not in job_titles

    def test_matches_stored_in_db(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        results = run_matching(conn, "ml-engineer", embedder=embedder)
        match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        assert match_count == len(results)

    def test_match_has_facet_scores(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        run_matching(conn, "ml-engineer", embedder=embedder)
        row = conn.execute("SELECT * FROM matches LIMIT 1").fetchone()
        facets = json.loads(row["facet_scores"])
        assert "skills" in facets
        assert "domain" in facets
        assert "seniority" in facets

    def test_job_extraction_cached(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        run_matching(conn, "ml-engineer", embedder=embedder)
        # Check that jobs now have extracted skills
        job = conn.execute("SELECT * FROM jobs WHERE source_id = 'j1'").fetchone()
        skills = json.loads(job["skills"])
        assert "python" in skills
        assert "pytorch" in skills

    def test_profile_embeddings_cached(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        run_matching(conn, "ml-engineer", embedder=embedder)
        profile = conn.execute("SELECT * FROM profiles WHERE name = 'ml-engineer'").fetchone()
        assert profile["skills_embedding"] is not None
        assert profile["domain_embedding"] is not None

    def test_rerun_updates_matches(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        results1 = run_matching(conn, "ml-engineer", embedder=embedder)
        results2 = run_matching(conn, "ml-engineer", embedder=embedder)
        # Same count — upsert, not duplicate
        match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        assert match_count == len(results1)

    def test_unknown_profile_raises(self, conn):
        embedder = FakeEmbedder()
        with pytest.raises(ValueError, match="Profile not found"):
            run_matching(conn, "nonexistent", embedder=embedder)

    def test_custom_weights(self, conn, sample_profile, sample_jobs):
        embedder = FakeEmbedder()
        weights = {"skills": 1.0, "domain": 0.0, "seniority": 0.0}
        results = run_matching(conn, "ml-engineer", embedder=embedder, weights=weights)
        assert len(results) > 0
        # All results should have seniority_score but it shouldn't affect ranking
        for r in results:
            assert "seniority_score" in r