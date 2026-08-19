"""SQLite database: schema creation, connection management, and helpers."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("jobscout.db")

# A line comment runs to end of line, so a semicolon written inside one is not
# a statement separator. Splitting first and discarding comment lines second
# cut a statement in half at such a semicolon and handed sqlite the remainder:
# a comment reading "...its score; what changes is..." produced
# `near "what": syntax error`. Comments come out before the split now.
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


def split_statements(migration_sql: str) -> list[str]:
    """Non-empty statements from a migration script, comments stripped first."""
    without_comments = _SQL_LINE_COMMENT_RE.sub("", migration_sql)
    return [
        statement
        for statement in (part.strip() for part in without_comments.split(";"))
        if statement
    ]

SCHEMA_SQL = """
-- Profiles derived from CVs. Multiple named profiles per install.
-- M2 will add: raw_text, skills, domains, seniority, languages, target_locations
-- M3 will add: embedding
CREATE TABLE IF NOT EXISTS profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,      -- e.g. "research-phd", "industry-mle"
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Job postings from any source.
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,              -- adapter name, e.g. "euraxess"
    source_id       TEXT,                          -- ID within that source
    url             TEXT,
    url_hash        TEXT,                          -- SHA-256 of URL for fast dedup
    title           TEXT    NOT NULL,
    company         TEXT,
    description     TEXT,
    location        TEXT,
    country         TEXT,                          -- ISO 3166-1 alpha-2
    language        TEXT,                          -- posting language (en/fr/de)
    seniority       TEXT,
    posted_at       TEXT,
    scraped_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    embedding       BLOB,                          -- reserved for M3
    raw_data        TEXT,                          -- original JSON blob

    UNIQUE(source, source_id)
);

-- Match scores between a profile and a job.
CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id      INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    job_id          INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    score           REAL    NOT NULL DEFAULT 0.0,  -- 0.0-1.0 composite
    facet_scores    TEXT    DEFAULT '{}',          -- JSON: {"skills": 0.8, "domain": 0.6, ...}
    explanation     TEXT    DEFAULT '[]',          -- JSON: ["MLOps", "Python", ...]
    status          TEXT,                          -- null / saved / applied / dismissed
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),

    UNIQUE(profile_id, job_id)
);

-- Scrape/scoring runs. HTMX polls this for progress.
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id      INTEGER REFERENCES profiles(id) ON DELETE SET NULL,
    status          TEXT    NOT NULL DEFAULT 'pending',  -- pending / running / completed / failed
    sources         TEXT    DEFAULT '[]',          -- JSON list of source names
    total_jobs      INTEGER DEFAULT 0,
    new_jobs        INTEGER DEFAULT 0,
    progress        REAL    DEFAULT 0.0,           -- 0.0-1.0
    started_at      TEXT,
    completed_at    TEXT,
    error           TEXT,

    CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    CHECK (progress >= 0.0 AND progress <= 1.0)
);

-- Indexes for the queries we'll actually run.
CREATE INDEX IF NOT EXISTS idx_jobs_url_hash      ON jobs(url_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_source         ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_matches_profile     ON matches(profile_id);
CREATE INDEX IF NOT EXISTS idx_matches_score       ON matches(score DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status         ON runs(status);
"""

# Add this constant after SCHEMA_SQL

# M2: Add profile columns for CV ingestion
MIGRATE_M2_SQL = """
ALTER TABLE profiles ADD COLUMN raw_text         TEXT;
ALTER TABLE profiles ADD COLUMN skills           TEXT DEFAULT '[]';
ALTER TABLE profiles ADD COLUMN domains          TEXT DEFAULT '[]';
ALTER TABLE profiles ADD COLUMN seniority        TEXT;
ALTER TABLE profiles ADD COLUMN languages        TEXT DEFAULT '[]';
ALTER TABLE profiles ADD COLUMN target_locations  TEXT DEFAULT '["Europe"]';
ALTER TABLE profiles ADD COLUMN company_types    TEXT DEFAULT '[]';
ALTER TABLE profiles ADD COLUMN position_types   TEXT DEFAULT '[]';
"""


def migrate_m2(conn: sqlite3.Connection) -> None:
    """Add M2 columns to profiles. Safe to call on already-migrated DBs."""
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(profiles)").fetchall()
    }
    m2_columns = [
        "raw_text", "skills", "domains", "seniority",
        "languages", "target_locations", "company_types", "position_types",
    ]
    if all(col in existing for col in m2_columns):
        return  # already migrated

    for statement in split_statements(MIGRATE_M2_SQL):
        col_name = statement.split("ADD COLUMN")[1].strip().split()[0]
        if col_name not in existing:
            conn.execute(statement)
    conn.commit()


# M3: Add embedding columns to profiles and job extraction columns to jobs
MIGRATE_M3_SQL = """
ALTER TABLE profiles ADD COLUMN skills_embedding BLOB;
ALTER TABLE profiles ADD COLUMN domain_embedding BLOB;
ALTER TABLE jobs ADD COLUMN skills           TEXT DEFAULT '[]';
ALTER TABLE jobs ADD COLUMN domains          TEXT DEFAULT '[]';
ALTER TABLE jobs ADD COLUMN skills_embedding  BLOB;
ALTER TABLE jobs ADD COLUMN domain_embedding  BLOB;
"""


def migrate_m3(conn: sqlite3.Connection) -> None:
    """Add M3 columns (embeddings + job extraction). Idempotent."""
    for statement in split_statements(MIGRATE_M3_SQL):
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    conn.commit()

def migrate_m4(conn: sqlite3.Connection) -> None:
    """Add M4 columns for cross-source dedup."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "dedup_hash" not in existing:
        conn.execute("ALTER TABLE jobs ADD COLUMN dedup_hash TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_dedup_hash ON jobs(dedup_hash)")
        conn.commit()

def migrate_m5(conn: sqlite3.Connection) -> None:
    """Add M5 columns: org_type on jobs, progress_message on runs."""
    # -- jobs.org_type --
    job_cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "org_type" not in job_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN org_type TEXT")
    
    # -- runs.progress_message --
    run_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "progress_message" not in run_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN progress_message TEXT")
    
    conn.commit()


def migrate_m6(conn: sqlite3.Connection) -> None:
    """Add position_type column to jobs."""
    cur = conn.execute("PRAGMA table_info(jobs)")
    cols = {row[1] for row in cur.fetchall()}
    if "position_type" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN position_type TEXT")
        conn.commit()


# M7b: Structured experience requirement, in years rather than buckets.
MIGRATE_M7B_SQL = """
ALTER TABLE jobs ADD COLUMN required_years_min REAL;
ALTER TABLE jobs ADD COLUMN education_level    TEXT;
ALTER TABLE jobs ADD COLUMN salary_yearly_min  INTEGER;
ALTER TABLE jobs ADD COLUMN salary_currency    TEXT;
ALTER TABLE jobs ADD COLUMN seniority_source   TEXT;
-- candidate_years is the authoritative figure and is only ever set by hand.
-- candidate_years_parsed is what CV date parsing suggested, kept for display.
ALTER TABLE profiles ADD COLUMN candidate_years        REAL;
ALTER TABLE profiles ADD COLUMN candidate_years_parsed REAL;
"""


def migrate_m7b(conn: sqlite3.Connection) -> None:
    """Add M7b columns for years-based seniority. Idempotent, per column."""
    _add_missing_columns(conn, MIGRATE_M7B_SQL)


def _add_missing_columns(conn: sqlite3.Connection, migration_sql: str) -> None:
    """Run the ALTER TABLE ... ADD COLUMN statements a table does not already have."""
    existing: dict[str, set[str]] = {}
    added = False

    for statement in split_statements(migration_sql):
        table = statement.split("ALTER TABLE")[1].strip().split()[0]
        col_name = statement.split("ADD COLUMN")[1].strip().split()[0]
        if table not in existing:
            existing[table] = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()  # noqa: S608
            }
        if col_name not in existing[table]:
            conn.execute(statement)
            existing[table].add(col_name)
            added = True

    if added:
        conn.commit()


# M7d: where a description came from, and whether the posting still exists.
MIGRATE_M7D_SQL = """
-- 'detail' = the posting's own description endpoint, 'blurb' = the search
-- index's requirements snippet, 'none' = neither yielded text.
ALTER TABLE jobs ADD COLUMN description_source TEXT;
-- Set when every attempt to reach the posting 404s. A delisted job keeps its
-- row and its score. What changes is only that its link is known dead.
ALTER TABLE jobs ADD COLUMN delisted_at TEXT;
"""


def migrate_m7d(conn: sqlite3.Connection) -> None:
    """Add M7d columns for description provenance. Idempotent, per column."""
    _add_missing_columns(conn, MIGRATE_M7D_SQL)


def find_by_dedup_hash(conn: sqlite3.Connection, dedup_hash: str) -> sqlite3.Row | None:
    """Return the first job matching this dedup hash, or None."""
    return conn.execute(
        "SELECT * FROM jobs WHERE dedup_hash = ?", (dedup_hash,)
    ).fetchone()


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a connection with sensible defaults (WAL, foreign keys, row factory)."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create the database and all tables. Idempotent."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def insert_job(conn: sqlite3.Connection, job: dict[str, Any]) -> int:
    """Insert a single job, returning its row id. Skips duplicates via IGNORE."""
    sql = """
        INSERT OR IGNORE INTO jobs
            (source, source_id, url, url_hash, title, company, description,
             location, country, language, seniority, posted_at, raw_data)
        VALUES
            (:source, :source_id, :url, :url_hash, :title, :company, :description,
             :location, :country, :language, :seniority, :posted_at, :raw_data)
    """
    cur = conn.execute(sql, job)
    conn.commit()
    return cur.lastrowid


def insert_jobs_bulk(conn: sqlite3.Connection, jobs: list[dict[str, Any]]) -> int:
    """Insert multiple jobs in a transaction. Returns count of rows inserted."""
    sql = """
        INSERT OR IGNORE INTO jobs
            (source, source_id, url, url_hash, title, company, description,
             location, country, language, seniority, posted_at, raw_data)
        VALUES
            (:source, :source_id, :url, :url_hash, :title, :company, :description,
             :location, :country, :language, :seniority, :posted_at, :raw_data)
    """
    before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    conn.executemany(sql, jobs)
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    return after - before


def get_all_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all jobs, most recent first."""
    return conn.execute("SELECT * FROM jobs ORDER BY scraped_at DESC").fetchall()


def get_job(conn: sqlite3.Connection, job_id: int) -> sqlite3.Row | None:
    """Return a single job by id."""
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Quick health check: row counts per table."""
    tables = ["profiles", "jobs", "matches", "runs"]
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
        for t in tables
    }

# Written on every upsert. candidate_years is deliberately not here: a
# hand-set override must survive re-ingesting the CV.
PROFILE_UPSERT_COLUMNS: tuple[str, ...] = (
    "raw_text", "skills", "domains", "seniority", "languages",
    "target_locations", "company_types", "position_types",
    "candidate_years_parsed",
)

PROFILE_JSON_COLUMNS: tuple[str, ...] = (
    "skills", "domains", "languages", "target_locations",
    "company_types", "position_types",
)


def upsert_profile(conn: sqlite3.Connection, profile: dict[str, Any]) -> int:
    """Insert or update a profile by name. Returns the profile id.

    The column list is intersected with what the table actually has, so a
    database that has not run every migration yet still writes what it can.
    """
    import json

    available = {
        row[1] for row in conn.execute("PRAGMA table_info(profiles)").fetchall()
    }
    cols = [c for c in PROFILE_UPSERT_COLUMNS if c in available]

    insert_cols = ", ".join(["name", *cols, "updated_at"])
    insert_vals = ", ".join([":name", *(f":{c}" for c in cols), "datetime('now')"])
    update_set = ", ".join(
        [*(f"{c} = excluded.{c}" for c in cols), "updated_at = datetime('now')"]
    )
    sql = (  # noqa: S608 — column names come from the module constant above
        f"INSERT INTO profiles ({insert_cols}) VALUES ({insert_vals}) "
        f"ON CONFLICT(name) DO UPDATE SET {update_set}"
    )

    params = {"name": profile["name"]}
    for col in cols:
        value = profile.get(col)
        if col in PROFILE_JSON_COLUMNS and isinstance(value, list):
            value = json.dumps(value)
        params[col] = value

    cur = conn.execute(sql, params)
    conn.commit()

    # Return the profile id
    row = conn.execute(
        "SELECT id FROM profiles WHERE name = ?", (params["name"],)
    ).fetchone()
    return row[0]


def get_profile(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    """Return a profile by name."""
    return conn.execute(
        "SELECT * FROM profiles WHERE name = ?", (name,)
    ).fetchone()


def get_all_profiles(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all profiles."""
    return conn.execute(
        "SELECT * FROM profiles ORDER BY created_at DESC"
    ).fetchall()