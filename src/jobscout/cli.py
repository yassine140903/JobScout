"""CLI entry point: jobscout init, ingest, profiles, match, fetch, serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from jobscout.config import load_config
from jobscout.db import (
    init_db, insert_jobs_bulk, table_counts,
    migrate_m2, migrate_m3, migrate_m4, migrate_m5,
    upsert_profile, get_profile, get_all_profiles,
)
from jobscout.fixtures import get_fixtures
from jobscout.matching import run_matching
from jobscout.profiles import extract_text, RuleBasedExtractor


def _setup_db(config: dict):
    """Init DB and run all migrations. Returns (conn, db_path)."""
    db_path = Path(config["db_path"])
    conn = init_db(db_path)
    migrate_m2(conn)
    migrate_m3(conn)
    migrate_m4(conn)
    migrate_m5(conn)
    return conn, db_path


def cmd_init(args: argparse.Namespace) -> None:
    """Create the database, load fixtures, and print a summary."""
    config = load_config(Path(args.config))
    conn, db_path = _setup_db(config)
    print(f"Initializing database at {db_path} ...")
    fixtures = get_fixtures()
    inserted = insert_jobs_bulk(conn, fixtures)
    print(f"Loaded {inserted} fixture postings ({len(fixtures)} total, duplicates skipped).")
    counts = table_counts(conn)
    for table, count in counts.items():
        print(f"  {table}: {count} rows")
    conn.close()
    print("Done.")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a CV and create/update a profile."""
    cv_path = Path(args.cv)
    if not cv_path.exists():
        print(f"Error: file not found: {cv_path}")
        sys.exit(1)

    config = load_config(Path(args.config))
    conn, _ = _setup_db(config)

    # Extract text
    print(f"Extracting text from {cv_path.name} ...")
    raw_text = extract_text(cv_path)
    if not raw_text.strip():
        print("Error: no text could be extracted from the file.")
        sys.exit(1)
    print(f"  Extracted {len(raw_text)} characters.")

    # Run extractor
    print("Analyzing profile ...")
    extractor = RuleBasedExtractor()
    facets = extractor.extract(raw_text)

    # Merge with CLI flags
    locations = [loc.strip() for loc in args.locations.split(",")] if args.locations else ["Europe"]
    company_types = [ct.strip() for ct in args.company_type.split(",")] if args.company_type else []
    position_types = [pt.strip() for pt in args.position_type.split(",")] if args.position_type else ["job"]

    profile = {
        "name": args.name,
        "raw_text": raw_text,
        "skills": facets["skills"],
        "domains": facets["domains"],
        "seniority": facets["seniority"],
        "languages": facets["languages"],
        "target_locations": locations,
        "company_types": company_types,
        "position_types": position_types,
    }

    profile_id = upsert_profile(conn, profile)

    # Print summary
    print(f"\nProfile '{args.name}' saved (id={profile_id}).")
    print(f"  Seniority:  {facets['seniority']}")
    print(f"  Skills:     {', '.join(facets['skills']) or '(none detected)'}")
    print(f"  Domains:    {', '.join(facets['domains']) or '(none detected)'}")
    print(f"  Languages:  {', '.join(facets['languages']) or '(none detected)'}")
    print(f"  Locations:  {', '.join(locations)}")
    print(f"  Company:    {', '.join(company_types) or '(any)'}")
    print(f"  Position:   {', '.join(position_types)}")

    conn.close()


def cmd_profiles(args: argparse.Namespace) -> None:
    """List all stored profiles."""
    config = load_config(Path(args.config))
    conn, _ = _setup_db(config)

    profiles = get_all_profiles(conn)
    if not profiles:
        print("No profiles yet. Run 'jobscout ingest' first.")
        conn.close()
        return

    for p in profiles:
        skills = json.loads(p["skills"]) if p["skills"] else []
        domains = json.loads(p["domains"]) if p["domains"] else []
        locations = json.loads(p["target_locations"]) if p["target_locations"] else []
        print(f"\n  {p['name']} (id={p['id']})")
        print(f"    seniority:  {p['seniority'] or '—'}")
        print(f"    skills:     {', '.join(skills) or '—'}")
        print(f"    domains:    {', '.join(domains) or '—'}")
        print(f"    locations:  {', '.join(locations)}")

    print()
    conn.close()


def cmd_match(args: argparse.Namespace) -> None:
    """Run matching for a profile against all jobs."""
    config = load_config(Path(args.config))
    conn, _ = _setup_db(config)

    profile = get_profile(conn, args.profile)
    if not profile:
        print(f"Error: profile '{args.profile}' not found.")
        conn.close()
        sys.exit(1)

    print(f"Matching profile '{args.profile}' against all jobs ...")
    results = run_matching(conn, args.profile)

    if not results:
        print("No matches found (check location/language filters).")
    else:
        print(f"\nTop {min(len(results), 20)} matches:\n")
        for i, r in enumerate(results[:20], 1):
            print(f"  {i:2d}. [{r['final_score']:.2f}] {r['job_title']} @ {r['job_company']}")
            print(f"      skills={r['skills_score']:.2f}  domain={r['domain_score']:.2f}  "
                  f"seniority×{r['seniority_multiplier']:.2f}")
            if r["matched_skills"]:
                print(f"      matched: {', '.join(r['matched_skills'])}")

    print(f"\n{len(results)} total matches stored.")
    conn.close()


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch jobs from configured sources."""
    from jobscout.sources import run_fetch

    config = load_config(Path(args.config))
    conn, _ = _setup_db(config)

    # Parse adapter filter
    adapter_names = None
    if args.sources:
        adapter_names = [s.strip() for s in args.sources.split(",")]

    # Set up logging so the user sees progress
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(name)s: %(message)s",
    )

    enabled = [
        s.get("name", s.get("adapter", "?"))
        for s in config.get("sources", [])
        if s.get("enabled", True)
    ]
    if adapter_names:
        print(f"Fetching from: {', '.join(adapter_names)}")
    else:
        print(f"Fetching from {len(enabled)} enabled sources: {', '.join(enabled)}")

    run_id = run_fetch(conn, config, adapter_names=adapter_names)

    if run_id < 0:
        print("No adapters enabled. Configure sources in config.yaml.")
        conn.close()
        sys.exit(1)

    # Print run summary
    run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run_row:
        print(f"\nRun {run_id}: {run_row['status']}")
        print(f"  Total fetched: {run_row['total_jobs']}")
        print(f"  New jobs:      {run_row['new_jobs']}")
        if run_row["error"]:
            print(f"  Errors:\n    {run_row['error']}")

    counts = table_counts(conn)
    print(f"  Jobs in DB:    {counts['jobs']}")
    conn.close()


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the web UI."""
    import uvicorn

    print(f"Starting JobScout on http://{args.host}:{args.port}")
    uvicorn.run(
        "jobscout.web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jobscout",
        description="Local-first, CV-driven job discovery tool.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    subparsers.add_parser("init", help="Create database and load fixtures")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a CV and create a profile")
    ingest_parser.add_argument("cv", help="Path to CV file (PDF or DOCX)")
    ingest_parser.add_argument("--name", required=True, help="Profile name (e.g. 'industry-mle')")
    ingest_parser.add_argument("--locations", default=None, help="Target locations, comma-separated")
    ingest_parser.add_argument("--company-type", default=None, help="Company types: startup,corporate,lab")
    ingest_parser.add_argument("--position-type", default=None, help="Position types: job,intern,phd,postdoc,freelance")

    # profiles
    subparsers.add_parser("profiles", help="List all stored profiles")

    # match
    match_parser = subparsers.add_parser("match", help="Run matching for a profile")
    match_parser.add_argument("profile", help="Profile name to match against")

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch jobs from configured sources")
    fetch_parser.add_argument("--sources", default=None, help="Comma-separated adapter names (default: all enabled)")
    fetch_parser.add_argument("-v", "--verbose", action="store_true", help="Show debug output")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the web UI")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev mode)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "profiles":
        cmd_profiles(args)
    elif args.command == "match":
        cmd_match(args)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()