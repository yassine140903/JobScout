"""Web UI: FastAPI + Jinja2 + HTMX."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, Depends, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from jobscout.config import load_config
from jobscout.db import (
    get_connection, init_db,
    migrate_m2, migrate_m3, migrate_m4, migrate_m5, migrate_m6,
    get_profile, upsert_profile,
)
from jobscout.profiles import extract_text, RuleBasedExtractor
from jobscout.sources import enrich_config_from_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

app = FastAPI(title="JobScout")
templates = Jinja2Templates(directory=str(_HERE / "templates"))
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def _get_db():
    """Yield a DB connection per request."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Startup: init + migrate
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup():
    conn = init_db()
    migrate_m2(conn)
    migrate_m3(conn)
    migrate_m4(conn)
    migrate_m5(conn)
    migrate_m6(conn)

    # Clean up fixture/dummy data
    deleted = conn.execute(
        "DELETE FROM jobs WHERE source = ?", ("fixture",)
    ).rowcount
    if deleted:
        conn.execute(
            "DELETE FROM matches WHERE job_id NOT IN (SELECT id FROM jobs)"
        )
        conn.commit()
        logger.info("Purged %d fixture jobs", deleted)

    from jobscout.sources import classify_org_type
    rows = conn.execute(
        "SELECT id, company, title, description FROM jobs WHERE org_type IS NULL"
    ).fetchall()
    for row in rows:
        org = classify_org_type(row["company"], row["title"], row["description"])
        conn.execute("UPDATE jobs SET org_type = ? WHERE id = ?", (org, row["id"]))
    if rows:
        conn.commit()
        logger.info("Backfilled org_type for %d jobs", len(rows))

    from jobscout.sources import classify_position_type
    rows = conn.execute(
        "SELECT id, title, description FROM jobs WHERE position_type IS NULL"
    ).fetchall()
    for row in rows:
        pt = classify_position_type(row["title"], row["description"])
        conn.execute("UPDATE jobs SET position_type = ? WHERE id = ?", (pt, row["id"]))
    if rows:
        conn.commit()
        logger.info("Backfilled position_type for %d jobs", len(rows))

    conn.close()
    logger.info("DB initialized and migrated")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload", response_class=HTMLResponse)
async def upload_cv(
    request: Request,
    cv: UploadFile = File(...),
    db=Depends(_get_db),
):
    """Upload a CV, extract a profile, return the profile card partial."""
    suffix = Path(cv.filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        return HTMLResponse(
            '<article class="error">Only .pdf and .docx files are supported.</article>',
            status_code=400,
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await cv.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        raw_text = extract_text(tmp_path)
        if not raw_text.strip():
            return HTMLResponse(
                '<article class="error">No text could be extracted from the file.</article>',
                status_code=400,
            )
        extractor = RuleBasedExtractor()
        result = extractor.extract(raw_text)

        profile_data = {
            "name": "default",
            "raw_text": raw_text,
            "skills": result.get("skills", []),
            "domains": result.get("domains", []),
            "seniority": result.get("seniority"),
            "languages": result.get("languages", []),
            "target_locations": [],
            "company_types": [],
            "position_types": [],
        }
        upsert_profile(db, profile_data)

        return templates.TemplateResponse(
            request, "partials/_profile_card.html",
            {"profile": profile_data, "filename": cv.filename},
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/runs", response_class=HTMLResponse)
async def start_run(
    request: Request,
    locations: str = Form(""),
    org_types: list[str] = Form(default=["lab", "startup", "corporate"]),
    db=Depends(_get_db),
):
    """Start a fetch + match pipeline in a background thread."""
    profile = get_profile(db, "default")
    if not profile:
        return HTMLResponse(
            '<article class="error">Upload a CV first.</article>',
            status_code=400,
        )

    location_list = [loc.strip() for loc in locations.split(",") if loc.strip()]
    db.execute(
        "UPDATE profiles SET target_locations = ? WHERE name = 'default'",
        (json.dumps(location_list),),
    )
    db.commit()

    cur = db.execute(
        "INSERT INTO runs (status, progress_message, started_at) "
        "VALUES ('running', 'Starting...', datetime('now'))",
    )
    db.commit()
    run_id = cur.lastrowid

    config = load_config()
    thread = threading.Thread(
        target=_run_pipeline,
        args=(run_id, config, org_types),
        daemon=True,
    )
    thread.start()

    return templates.TemplateResponse(
        request, "partials/_run_status.html",
        {
            "run_id": run_id,
            "progress_message": "Starting...",
            "progress": 0.0,
            "status": "running",
        },
    )


@app.get("/runs/{run_id}/status", response_class=HTMLResponse)
def run_status(request: Request, run_id: int, db=Depends(_get_db)):
    """Return the current run status partial (HTMX polls this)."""
    run = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        return HTMLResponse("Run not found", status_code=404)

    return templates.TemplateResponse(
        request, "partials/_run_status.html",
        {
            "run_id": run_id,
            "progress_message": run["progress_message"] or "",
            "progress": run["progress"] or 0.0,
            "status": run["status"],
        },
    )


@app.get("/jobs", response_class=HTMLResponse)
def list_jobs(
    request: Request,
    profile: str = "default",
    min_score: float = 0.0,
    source: str = "",
    location: str = "",
    org_type: str = "",
    position_type: str = "",
    db=Depends(_get_db),
):
    """Return the jobs table partial, with optional filters."""
    query = """
        SELECT j.*, m.score, m.facet_scores, m.explanation, m.status AS match_status
        FROM jobs j
        JOIN matches m ON m.job_id = j.id
        JOIN profiles p ON p.id = m.profile_id
        WHERE p.name = ?
          AND m.score >= ?
    """
    params: list = [profile, min_score]

    if source:
        query += " AND j.source = ?"
        params.append(source)
    if location:
        query += " AND (LOWER(j.location) LIKE ? OR LOWER(j.country) LIKE ?)"
        params.extend([f"%{location.lower()}%", f"%{location.lower()}%"])
    if org_type:
        query += " AND j.org_type = ?"
        params.append(org_type)
    if position_type:
        query += " AND j.position_type = ?"
        params.append(position_type)

    query += " ORDER BY m.score DESC"

    jobs = db.execute(query, params).fetchall()

    sources = db.execute(
        "SELECT DISTINCT source FROM jobs ORDER BY source"
    ).fetchall()
    locations = db.execute(
        "SELECT DISTINCT location FROM jobs WHERE location IS NOT NULL "
        "ORDER BY location"
    ).fetchall()

    return templates.TemplateResponse(
        request, "partials/_jobs_table.html",
        {
            "jobs": jobs,
            "sources": [r["source"] for r in sources],
            "locations": [r["location"] for r in locations],
            "current_filters": {
                "min_score": min_score,
                "source": source,
                "location": location,
                "org_type": org_type,
                "position_type": position_type,
            },
        },
    )


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int, db=Depends(_get_db)):
    """Return the job detail partial with score breakdown."""
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        return HTMLResponse("Job not found", status_code=404)

    match = db.execute(
        "SELECT * FROM matches WHERE job_id = ? AND profile_id = "
        "(SELECT id FROM profiles WHERE name = 'default')",
        (job_id,),
    ).fetchone()

    facet_scores = {}
    matched_skills = []
    if match:
        facet_scores = json.loads(match["facet_scores"] or "{}")
        matched_skills = json.loads(match["explanation"] or "[]")

    return templates.TemplateResponse(
        request, "partials/_job_detail.html",
        {
            "job": job,
            "match": match,
            "facet_scores": facet_scores,
            "matched_skills": matched_skills,
        },
    )


@app.post("/jobs/{job_id}/status", response_class=HTMLResponse)
def update_job_status(
    request: Request,
    job_id: int,
    status: str = Form(...),
    db=Depends(_get_db),
):
    """Update the match status (saved/applied/dismissed)."""
    current = db.execute(
        "SELECT status FROM matches WHERE job_id = ? AND profile_id = "
        "(SELECT id FROM profiles WHERE name = 'default')",
        (job_id,),
    ).fetchone()

    new_status = None if current and current["status"] == status else status

    db.execute(
        "UPDATE matches SET status = ? WHERE job_id = ? AND profile_id = "
        "(SELECT id FROM profiles WHERE name = 'default')",
        (new_status, job_id),
    )
    db.commit()

    return templates.TemplateResponse(
        request, "partials/_status_buttons.html",
        {"job_id": job_id, "current_status": new_status},
    )


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(run_id: int, config: dict, org_types: list[str] | None = None):
    """Background: fetch jobs → score against profile. Runs in its own thread."""
    conn = get_connection()
    try:
        # --- Enrich config from profile ---
        profile = get_profile(conn, "default")
        if profile:
            config = enrich_config_from_profile(config, profile)

        # Build adapter name list for progress messages
        enabled = [
            s.get("name", s.get("adapter", "?"))
            for s in config.get("sources", [])
            if s.get("enabled", True)
        ]
        adapter_label = ", ".join(enabled) if enabled else "no sources"

        _update_run(conn, run_id, progress=0.1,
                     message=f"Fetching from {adapter_label}...")

        from jobscout.sources import run_fetch
        fetch_run_id = run_fetch(conn, config)

        if fetch_run_id < 0:
            _update_run(conn, run_id, progress=0.3,
                         message="No adapters returned results — scoring existing jobs...")
        else:
            # Summarize fetch results
            fetch_row = conn.execute(
                "SELECT total_jobs, new_jobs, error FROM runs WHERE id = ?",
                (fetch_run_id,),
            ).fetchone()
            if fetch_row and fetch_row["error"]:
                logger.warning("Fetch had errors: %s", fetch_row["error"])

        job_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        _update_run(conn, run_id, progress=0.5,
                     message="Scoring jobs against profile (loading model, this may take a moment)...")

        from jobscout.matching import run_matching
        weights = config.get("scoring", {}).get("weights")
        results = run_matching(conn, "default", weights=weights, org_types=org_types)

        conn.execute(
            "UPDATE runs SET status = 'completed', progress = 1.0, "
            "progress_message = ?, total_jobs = ?, new_jobs = ?, "
            "completed_at = datetime('now') WHERE id = ?",
            (f"Done — {len(results)} matches found", job_count, len(results), run_id),
        )
        conn.commit()

    except Exception as exc:
        logger.exception("Pipeline failed")
        conn.execute(
            "UPDATE runs SET status = 'failed', "
            "progress_message = ?, error = ?, completed_at = datetime('now') "
            "WHERE id = ?",
            (f"Failed: {exc}", str(exc), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _update_run(conn, run_id: int, progress: float, message: str):
    """Update run progress and message."""
    conn.execute(
        "UPDATE runs SET progress = ?, progress_message = ? WHERE id = ?",
        (progress, message, run_id),
    )
    conn.commit()