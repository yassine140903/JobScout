"""Tests for M4: source adapters, normalization, dedup, orchestrator."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from jobscout.db import (
    init_db, migrate_m2, migrate_m3, migrate_m4, migrate_m5, migrate_m6, migrate_m7b,
    migrate_m7d,
    find_by_dedup_hash,
)
from jobscout.sources import (
    RawPosting,
    SourceAdapter,
    compute_dedup_hash,
    compute_url_hash,
    normalize,
    run_fetch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    """Fresh DB with all migrations applied."""
    conn = init_db(tmp_path / "test.db")
    migrate_m2(conn)
    migrate_m3(conn)
    migrate_m4(conn)
    migrate_m5(conn)
    migrate_m6(conn)
    migrate_m7b(conn)
    migrate_m7d(conn)
    return conn


def _make_posting(**overrides) -> RawPosting:
    """Create a RawPosting with sensible defaults."""
    defaults = {
        "title": "ML Engineer",
        "source": "test",
        "company": "Acme Corp",
        "url": "https://example.com/jobs/1",
        "description": "Build ML pipelines.",
        "location": "Paris",
        "country": "FR",
        "language": "en",
        "seniority": "mid",
        "posted_at": "2026-08-01",
        "source_id": "test-001",
    }
    defaults.update(overrides)
    return RawPosting(**defaults)


class FakeAdapter(SourceAdapter):
    """Test adapter that returns pre-set postings."""

    def __init__(self, name: str, postings: list[RawPosting] | None = None, error: Exception | None = None):
        self.name = name
        self._postings = postings or []
        self._error = error
        self.fetch_count = 0

    def fetch(self, source_config: dict) -> list[RawPosting]:
        self.fetch_count += 1
        if self._error:
            raise self._error
        return self._postings


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

class TestMigrateM4:
    def test_adds_dedup_hash_column(self, db: sqlite3.Connection):
        cols = {row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "dedup_hash" in cols

    def test_idempotent(self, db: sqlite3.Connection):
        migrate_m4(db)  # second call should not raise
        cols = {row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "dedup_hash" in cols

    def test_dedup_hash_index_exists(self, db: sqlite3.Connection):
        indexes = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='jobs'"
        ).fetchall()
        index_names = {row[0] for row in indexes}
        assert "idx_jobs_dedup_hash" in index_names


# ---------------------------------------------------------------------------
# Dedup hashing
# ---------------------------------------------------------------------------

class TestDedupHash:
    def test_same_title_company(self):
        h1 = compute_dedup_hash("ML Engineer", "Acme Corp")
        h2 = compute_dedup_hash("ML Engineer", "Acme Corp")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = compute_dedup_hash("ML Engineer", "Acme Corp")
        h2 = compute_dedup_hash("ml engineer", "acme corp")
        assert h1 == h2

    def test_whitespace_stripped(self):
        h1 = compute_dedup_hash("ML Engineer", "Acme Corp")
        h2 = compute_dedup_hash("  ML Engineer  ", "  Acme Corp  ")
        assert h1 == h2

    def test_different_title_different_hash(self):
        h1 = compute_dedup_hash("ML Engineer", "Acme Corp")
        h2 = compute_dedup_hash("Data Scientist", "Acme Corp")
        assert h1 != h2

    def test_different_company_different_hash(self):
        h1 = compute_dedup_hash("ML Engineer", "Acme Corp")
        h2 = compute_dedup_hash("ML Engineer", "Other Corp")
        assert h1 != h2

    def test_none_company(self):
        h1 = compute_dedup_hash("ML Engineer", None)
        h2 = compute_dedup_hash("ML Engineer", None)
        assert h1 == h2

    def test_url_hash_returns_none_for_none(self):
        assert compute_url_hash(None) is None

    def test_url_hash_deterministic(self):
        h1 = compute_url_hash("https://example.com/jobs/1")
        h2 = compute_url_hash("https://example.com/jobs/1")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic_fields(self):
        raw = _make_posting()
        job = normalize(raw)
        assert job["title"] == "ML Engineer"
        assert job["source"] == "test"
        assert job["company"] == "Acme Corp"
        assert job["country"] == "FR"
        assert job["language"] == "en"

    def test_url_hash_computed(self):
        raw = _make_posting(url="https://example.com/jobs/1")
        job = normalize(raw)
        assert job["url_hash"] is not None
        assert len(job["url_hash"]) == 64

    def test_dedup_hash_computed(self):
        raw = _make_posting()
        job = normalize(raw)
        assert job["dedup_hash"] is not None
        assert job["dedup_hash"] == compute_dedup_hash("ML Engineer", "Acme Corp")

    def test_raw_data_serialized(self):
        raw = _make_posting(raw_data={"key": "value"})
        job = normalize(raw)
        assert json.loads(job["raw_data"]) == {"key": "value"}

    def test_raw_data_none(self):
        raw = _make_posting(raw_data=None)
        job = normalize(raw)
        assert job["raw_data"] is None

    def test_no_url_no_hash(self):
        raw = _make_posting(url=None)
        job = normalize(raw)
        assert job["url_hash"] is None


# ---------------------------------------------------------------------------
# find_by_dedup_hash
# ---------------------------------------------------------------------------

class TestFindByDedupHash:
    def test_finds_existing(self, db: sqlite3.Connection):
        dh = compute_dedup_hash("ML Engineer", "Acme Corp")
        db.execute(
            "INSERT INTO jobs (source, source_id, title, company, dedup_hash) "
            "VALUES ('test', 't1', 'ML Engineer', 'Acme Corp', ?)",
            (dh,),
        )
        db.commit()
        row = find_by_dedup_hash(db, dh)
        assert row is not None
        assert row["title"] == "ML Engineer"

    def test_returns_none_for_missing(self, db: sqlite3.Connection):
        assert find_by_dedup_hash(db, "nonexistent") is None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestRunFetch:
    def _config_with_adapters(self, *adapter_names: str) -> dict:
        return {
            "db_path": "test.db",
            "sources": [
                {"name": name, "adapter": name, "enabled": True}
                for name in adapter_names
            ],
        }

    @patch("jobscout.sources._get_adapters")
    def test_basic_fetch(self, mock_get, db: sqlite3.Connection):
        postings = [_make_posting(source_id="j1"), _make_posting(title="Data Scientist", source_id="j2")]
        adapter = FakeAdapter("test_src", postings=postings)
        mock_get.return_value = [(adapter, {"name": "test_src"})]

        run_id = run_fetch(db, {})
        assert run_id > 0

        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["status"] == "completed"
        assert run_row["new_jobs"] == 2
        assert run_row["total_jobs"] == 2

    @patch("jobscout.sources._get_adapters")
    def test_cross_source_dedup(self, mock_get, db: sqlite3.Connection):
        """Same title+company from two adapters → only one inserted."""
        p1 = _make_posting(source="src_a", source_id="a1")
        p2 = _make_posting(source="src_b", source_id="b1")  # same title+company
        a1 = FakeAdapter("src_a", postings=[p1])
        a2 = FakeAdapter("src_b", postings=[p2])
        mock_get.return_value = [(a1, {}), (a2, {})]

        run_id = run_fetch(db, {})
        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["new_jobs"] == 1

    @patch("jobscout.sources._get_adapters")
    def test_db_dedup_across_runs(self, mock_get, db: sqlite3.Connection):
        """Second run with same postings → zero new jobs."""
        postings = [_make_posting()]
        adapter = FakeAdapter("test_src", postings=postings)
        mock_get.return_value = [(adapter, {})]

        run_fetch(db, {})
        run_id = run_fetch(db, {})

        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["new_jobs"] == 0

    @patch("jobscout.sources._get_adapters")
    def test_failure_isolation(self, mock_get, db: sqlite3.Connection):
        """One adapter fails, other still succeeds."""
        good = FakeAdapter("good", postings=[_make_posting(source="good", source_id="g1")])
        bad = FakeAdapter("bad", error=ValueError("parser broke"))
        mock_get.return_value = [(bad, {}), (good, {})]

        run_id = run_fetch(db, {})
        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["status"] == "completed"  # partial success
        assert run_row["new_jobs"] == 1
        assert "bad: parser broke" in run_row["error"]

    @patch("jobscout.sources._get_adapters")
    def test_all_fail(self, mock_get, db: sqlite3.Connection):
        bad1 = FakeAdapter("bad1", error=RuntimeError("fail1"))
        bad2 = FakeAdapter("bad2", error=RuntimeError("fail2"))
        mock_get.return_value = [(bad1, {}), (bad2, {})]

        run_id = run_fetch(db, {})
        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["status"] == "failed"

    @patch("jobscout.sources._get_adapters")
    def test_no_adapters_returns_negative(self, mock_get, db: sqlite3.Connection):
        mock_get.return_value = []
        assert run_fetch(db, {}) == -1

    @patch("jobscout.sources._get_adapters")
    def test_progress_updated(self, mock_get, db: sqlite3.Connection):
        a1 = FakeAdapter("a", postings=[_make_posting(source="a", source_id="a1")])
        a2 = FakeAdapter("b", postings=[_make_posting(source="b", source_id="b1", title="Other Role")])
        mock_get.return_value = [(a1, {}), (a2, {})]

        run_id = run_fetch(db, {})
        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["progress"] == 1.0

    @patch("jobscout.sources._get_adapters")
    def test_adapter_name_filter(self, mock_get, db: sqlite3.Connection):
        a1 = FakeAdapter("wanted", postings=[_make_posting(source="wanted", source_id="w1")])
        a2 = FakeAdapter("skipped", postings=[_make_posting(source="skipped", source_id="s1")])
        mock_get.return_value = [(a1, {}), (a2, {})]

        run_id = run_fetch(db, {}, adapter_names=["wanted"])
        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["new_jobs"] == 1


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetry:
    @patch("jobscout.sources._get_adapters")
    def test_retry_on_network_error(self, mock_get, db: sqlite3.Connection):
        """Adapter fails once with network error, succeeds on retry."""
        import httpx

        call_count = 0
        postings = [_make_posting(source="flaky", source_id="f1")]

        class FlakyAdapter(SourceAdapter):
            name = "flaky"

            def fetch(self, source_config):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise httpx.ConnectError("connection refused")
                return postings

        mock_get.return_value = [(FlakyAdapter(), {})]

        run_id = run_fetch(db, {})
        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["status"] == "completed"
        assert run_row["new_jobs"] == 1
        assert call_count == 2

    @patch("jobscout.sources._get_adapters")
    def test_gives_up_after_retry(self, mock_get, db: sqlite3.Connection):
        """Adapter fails twice (initial + retry) → recorded as error."""
        import httpx

        class AlwaysDownAdapter(SourceAdapter):
            name = "down"

            def fetch(self, source_config):
                raise httpx.TimeoutException("timed out")

        mock_get.return_value = [(AlwaysDownAdapter(), {})]

        run_id = run_fetch(db, {})
        run_row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert run_row["status"] == "failed"
        assert "timed out" in run_row["error"]


# ---------------------------------------------------------------------------
# WTTJ adapter — parsing logic (no network)
# ---------------------------------------------------------------------------

class TestWTTJParsing:
    def test_hit_to_posting(self):
        from jobscout.sources.wttj import WTTJAdapter

        hit = {
            "name": "Senior ML Engineer",
            "objectID": "12345",
            "slug": "senior-ml-engineer_paris",
            "organization": {
                "name": "DeepTech SAS",
                "slug": "deeptech",
            },
            "office": {
                "city": "Paris",
                "country_code": "FR",
            },
            "published_at": "2026-08-01T00:00:00.000+02:00",
            "contract_type_names": {"en": "Full-Time"},
            "has_experience_level_minimum": True,
            "experience_level_minimum": 3,
            "language": "en",
            # M7d: `profile` is a flat string on every hit in this index, and
            # `recruitment_process` is not a key at all. This fixture used to
            # assert the per-language dict shape the adapter defended against
            # — a defence for a schema WTTJ has never served.
            "profile": "We are looking for a senior ML engineer.",
        }

        posting = WTTJAdapter._hit_to_posting(hit)
        assert posting is not None
        assert posting.title == "Senior ML Engineer"
        assert posting.company == "DeepTech SAS"
        assert posting.country == "FR"
        assert posting.location == "Paris"
        assert posting.source == "wttj"
        assert posting.source_id == "12345"
        # M7b: years, not a bucket. The bucket field is no longer guessed at.
        assert posting.required_years_min == 3.0
        assert posting.seniority_source == "api"
        assert posting.seniority is None
        assert "senior ML engineer" in posting.description
        # The blurb is what the index gives; the detail fetch overwrites it.
        assert posting.description_source == "blurb"
        assert "https://www.welcometothejungle.com/en/companies/deeptech/jobs/senior-ml-engineer_paris" == posting.url

    def test_hit_missing_title_returns_none(self):
        from jobscout.sources.wttj import WTTJAdapter

        assert WTTJAdapter._hit_to_posting({}) is None
        assert WTTJAdapter._hit_to_posting({"name": ""}) is None


# ---------------------------------------------------------------------------
# EURES adapter — parsing logic (no network)
# ---------------------------------------------------------------------------

class TestEURESParsing:
    def test_item_to_posting(self):
        from jobscout.sources.eures import EURESAdapter

        item = {
            "id": "abc123",
            "title": "Data Scientist",
            "employer": "EU Research Lab",
            "description": "Analyze large datasets for policy insights.",
            "locationMap": {
                "fr": [{"label": "Paris"}, {"label": "Lyon"}],
            },
            "publicationDate": "2026-07-15",
            "language": "en",
            "requiredExperienceMonths": 36,
        }

        posting = EURESAdapter._item_to_posting(item)
        assert posting is not None
        assert posting.title == "Data Scientist"
        assert posting.company == "EU Research Lab"
        assert posting.country == "FR"
        assert posting.location == "Paris, Lyon"
        assert posting.seniority == "mid"
        assert posting.source == "eures"
        assert posting.source_id == "abc123"
        assert "europa.eu/eures" in posting.url

    def test_item_missing_title_returns_none(self):
        from jobscout.sources.eures import EURESAdapter

        assert EURESAdapter._item_to_posting({}) is None

    def test_experience_mapping(self):
        from jobscout.sources.eures import _map_experience

        assert _map_experience(3) == "intern"
        assert _map_experience(18) == "junior"
        assert _map_experience(36) == "mid"
        assert _map_experience(84) == "senior"
        assert _map_experience(150) == "lead"
        assert _map_experience(None) is None


# ---------------------------------------------------------------------------
# EURAXESS adapter — seniority mapping
# ---------------------------------------------------------------------------

class TestEURAXESSMapping:
    def test_researcher_profile_mapping(self):
        from jobscout.sources.euraxess import _map_researcher_profile

        assert _map_researcher_profile("First Stage Researcher (R1)") == "junior"
        assert _map_researcher_profile("Recognised Researcher (R2)") == "mid"
        assert _map_researcher_profile("Established Researcher (R3)") == "senior"
        assert _map_researcher_profile("Leading Researcher (R4)") == "lead"
        assert _map_researcher_profile("Other Profession") is None
        assert _map_researcher_profile(None) is None

    def test_country_mapping(self):
        from jobscout.sources.euraxess import _EURAXESS_COUNTRIES

        assert _EURAXESS_COUNTRIES["France"] == "FR"
        assert _EURAXESS_COUNTRIES["Germany"] == "DE"
        assert _EURAXESS_COUNTRIES["Tunisia"] == "TN"


# ---------------------------------------------------------------------------
# Generic RSS adapter — parsing logic (no network)
# ---------------------------------------------------------------------------

class TestGenericRSSParsing:
    def test_entry_to_posting(self):
        from jobscout.sources.generic_rss import GenericRSSAdapter, DEFAULT_FIELD_MAP

        entry = {
            "title": "Backend Developer",
            "author": "TechCo",
            "link": "https://example.com/jobs/backend",
            "summary": "Build <b>REST APIs</b> with Python.",
            "published": "2026-08-10",
            "id": "job-42",
        }

        posting = GenericRSSAdapter._entry_to_posting(
            entry, DEFAULT_FIELD_MAP, "test_feed", "DE", "en", "Berlin",
        )
        assert posting is not None
        assert posting.title == "Backend Developer"
        assert posting.company == "TechCo"
        assert posting.url == "https://example.com/jobs/backend"
        assert "REST APIs" in posting.description
        assert "<b>" not in posting.description  # HTML stripped
        assert posting.country == "DE"
        assert posting.language == "en"
        assert posting.source == "test_feed"

    def test_custom_field_map(self):
        from jobscout.sources.generic_rss import GenericRSSAdapter

        entry = {
            "title": "Researcher",
            "dc_creator": "University Lab",
            "link": "https://uni.edu/job/1",
            "content": [{"value": "Full description here."}],
            "updated": "2026-08-12",
            "id": "r-1",
        }
        custom_map = {
            "title": "title",
            "company": "dc_creator",
            "url": "link",
            "description": "summary",  # intentionally wrong — should fall back to content
            "posted_at": "updated",
            "source_id": "id",
        }

        posting = GenericRSSAdapter._entry_to_posting(
            entry, custom_map, "uni_feed", None, None, None,
        )
        assert posting is not None
        assert posting.title == "Researcher"
        assert posting.company == "University Lab"
        assert posting.description == "Full description here."
        assert posting.posted_at == "2026-08-12"

    def test_missing_title_returns_none(self):
        from jobscout.sources.generic_rss import GenericRSSAdapter, DEFAULT_FIELD_MAP

        posting = GenericRSSAdapter._entry_to_posting(
            {"summary": "no title"}, DEFAULT_FIELD_MAP, "x", None, None, None,
        )
        assert posting is None

    def test_nested_field_map(self):
        from jobscout.sources.generic_rss import _resolve_nested

        data = {"author": {"name": "Lab X"}}
        assert _resolve_nested(data, "author.name") == "Lab X"
        assert _resolve_nested(data, "author.email") is None
        assert _resolve_nested(data, "missing.field") is None

# ---------------------------------------------------------------------------
# Profile-driven config enrichment
# ---------------------------------------------------------------------------

def test_enrich_config_from_profile_fills_empty_keywords():
    """enrich_config_from_profile injects profile skills/domains as keywords."""
    from jobscout.sources import enrich_config_from_profile
    config = {
        "sources": [
            {"name": "wttj", "adapter": "wttj", "enabled": True, "keywords": []},
            {"name": "eures", "adapter": "eures", "enabled": True, "keywords": [], "locations": []},
        ]
    }
    profile = {
        "skills": json.dumps(["Python", "Docker", "TensorFlow"]),
        "domains": json.dumps(["AI", "MLOps"]),
        "target_locations": json.dumps(["Paris", "Berlin"]),
    }
    result = enrich_config_from_profile(config, profile)
    assert result["sources"][0]["keywords"] == ["AI", "MLOps", "Python", "Docker", "TensorFlow"]
    assert result["sources"][1]["keywords"] == ["AI", "MLOps", "Python", "Docker", "TensorFlow"]
    assert result["sources"][1]["locations"] == []
    # Original not mutated
    assert config["sources"][0]["keywords"] == []


def test_enrich_config_preserves_existing_keywords():
    """enrich_config_from_profile does not overwrite user-configured keywords."""
    from jobscout.sources import enrich_config_from_profile
    config = {
        "sources": [
            {"name": "wttj", "adapter": "wttj", "enabled": True, "keywords": ["custom"]},
        ]
    }
    profile = {
        "skills": json.dumps(["Python"]),
        "domains": json.dumps(["AI"]),
        "target_locations": json.dumps([]),
    }
    result = enrich_config_from_profile(config, profile)
    assert result["sources"][0]["keywords"] == ["custom"]
