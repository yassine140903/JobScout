"""M1 tests: schema creation, fixture loading, config, idempotency."""

from pathlib import Path
import tempfile

from jobscout.config import load_config, DEFAULTS
from jobscout.db import init_db, insert_jobs_bulk, get_all_jobs, get_job, table_counts
from jobscout.fixtures import get_fixtures
from jobscout.matching import DEFAULT_WEIGHTS


def _temp_db() -> Path:
    """Return a fresh temp DB path."""
    return Path(tempfile.mktemp(suffix=".db"))


def test_schema_creates_all_tables():
    db = _temp_db()
    conn = init_db(db)
    counts = table_counts(conn)
    assert set(counts.keys()) == {"profiles", "jobs", "matches", "runs"}
    for count in counts.values():
        assert count == 0
    conn.close()
    db.unlink()


def test_fixtures_load():
    fixtures = get_fixtures()
    assert len(fixtures) == 20
    for f in fixtures:
        assert f["url_hash"] is not None
        assert f["source"] == "fixture"


def test_insert_and_query():
    db = _temp_db()
    conn = init_db(db)
    fixtures = get_fixtures()

    inserted = insert_jobs_bulk(conn, fixtures)
    assert inserted == 20

    jobs = get_all_jobs(conn)
    assert len(jobs) == 20

    job = get_job(conn, 1)
    assert job is not None
    assert job["source"] == "fixture"

    conn.close()
    db.unlink()


def test_idempotent_init():
    db = _temp_db()
    conn = init_db(db)
    insert_jobs_bulk(conn, get_fixtures())

    # second init + insert should not duplicate
    conn2 = init_db(db)
    inserted = insert_jobs_bulk(conn2, get_fixtures())
    assert inserted == 0
    assert table_counts(conn2)["jobs"] == 20

    conn.close()
    conn2.close()
    db.unlink()


def test_config_defaults():
    config = load_config(Path("nonexistent.yaml"))
    assert config == DEFAULTS
    assert config["db_path"] == "jobscout.db"
    # Config defaults must agree with the scorer's own defaults
    assert config["scoring"]["weights"] == DEFAULT_WEIGHTS
    assert config["scoring"]["weights"]["skills"] == 0.60
    assert config["scoring"]["weights"]["domain"] == 0.40
    # Seniority is a multiplier now, not a weighted facet
    assert "seniority" not in config["scoring"]["weights"]


def test_mixed_languages_in_fixtures():
    fixtures = get_fixtures()
    languages = {f["language"] for f in fixtures}
    assert "en" in languages
    assert "fr" in languages


def test_mixed_seniority_in_fixtures():
    fixtures = get_fixtures()
    levels = {f["seniority"] for f in fixtures}
    assert levels >= {"junior", "mid", "senior"}