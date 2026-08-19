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
    migrate_m2, migrate_m3, migrate_m4, migrate_m5, migrate_m6, migrate_m7b,
    migrate_m7d,
    upsert_profile, get_profile, get_all_profiles,
)
from jobscout.fixtures import get_fixtures
from jobscout.matching import (
    DEFAULT_FILTER_ON_INFERRED, DEFAULT_GATE_YEARS, run_matching,
)
from jobscout.profiles import (
    extract_text, parse_experience_years, resolve_candidate_years,
    RuleBasedExtractor,
)

logger = logging.getLogger(__name__)


def _format_years(years: float | None, source: str) -> str:
    """Render the candidate's years plus where the number came from."""
    if years is None:
        return "not set (seniority scoring stays neutral)"
    return f"{years:g} years (from {source})"


def _setup_db(config: dict):
    """Init DB and run all migrations. Returns (conn, db_path)."""
    db_path = Path(config["db_path"])
    conn = init_db(db_path)
    migrate_m2(conn)
    migrate_m3(conn)
    migrate_m4(conn)
    migrate_m5(conn)
    migrate_m6(conn)
    migrate_m7b(conn)
    migrate_m7d(conn)
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
    x_tolerance = config.get("extraction", {}).get("pdf_x_tolerance")
    raw_text = extract_text(cv_path, x_tolerance=x_tolerance)
    if not raw_text.strip():
        print("Error: no text could be extracted from the file.")
        sys.exit(1)
    print(f"  Extracted {len(raw_text)} characters.")

    # Run extractor
    print("Analyzing profile ...")
    extractor = RuleBasedExtractor()
    facets = extractor.extract(raw_text)
    parsed_years = parse_experience_years(raw_text)

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
        "candidate_years_parsed": parsed_years.years,
    }

    profile_id = upsert_profile(conn, profile)

    years, years_source = resolve_candidate_years(get_profile(conn, args.name), config)

    # Print summary
    print(f"\nProfile '{args.name}' saved (id={profile_id}).")
    print(f"  Seniority:  {facets['seniority']}")
    print(f"  Experience: {_format_years(years, years_source)}")
    print(f"    CV dates suggest {parsed_years.years:.2f} years"
          f"{' — ' + ', '.join(parsed_years.ranges) if parsed_years.ranges else ' (no date ranges found)'}")
    if parsed_years.ignored:
        print(f"    {len(parsed_years.ignored)} date-bearing line(s) not parsed, "
              f"so not counted:")
        for line in parsed_years.ignored[:5]:
            print(f"      - {line[:92]}")
    if years_source == "unset":
        print("    Set profile.candidate_years in config.yaml to score on it.")
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
        years, years_source = resolve_candidate_years(p, config)
        print(f"\n  {p['name']} (id={p['id']})")
        print(f"    seniority:  {p['seniority'] or '—'}")
        print(f"    experience: {_format_years(years, years_source)}")
        if p["candidate_years_parsed"] is not None:
            print(f"      CV dates suggest {p['candidate_years_parsed']:g} years "
                  f"(advisory — not used unless you set it)")
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

    scoring = config.get("scoring", {})
    weights = scoring.get("weights")
    seniority_cfg = scoring.get("seniority") or {}
    gate = seniority_cfg.get("gate_years", DEFAULT_GATE_YEARS)
    filter_on_inferred = seniority_cfg.get(
        "filter_on_inferred", DEFAULT_FILTER_ON_INFERRED,
    )
    years, years_source = resolve_candidate_years(profile, config)

    print(f"Matching profile '{args.profile}' against all jobs ...")
    print(f"  Experience: {_format_years(years, years_source)}  "
          f"gate: {gate:g} years  "
          f"filter on inferred: {'yes' if filter_on_inferred else 'no'}")

    results = run_matching(
        conn, args.profile, weights=weights,
        candidate_years=years, gate_years=gate,
        filter_on_inferred=filter_on_inferred,
    )

    stretch = [r for r in results if r["filtered"]]
    shown = results if args.show_stretch else [r for r in results if not r["filtered"]]

    if not shown:
        print("No matches found (check location/language filters).")
    else:
        print(f"\nTop {min(len(shown), 20)} matches:\n")
        for i, r in enumerate(shown[:20], 1):
            sen = r["seniority"]
            flag = "  [STRETCH]" if r["filtered"] else ""
            print(f"  {i:2d}. [{r['final_score']:.2f}] {r['job_title']} @ {r['job_company']}{flag}")
            print(f"      skills={r['skills_score']:.2f}  domain={r['domain_score']:.2f}  "
                  f"seniority×{r['seniority_multiplier']:.2f}  {_seniority_note(sen)}")
            if r["matched_skills"]:
                print(f"      matched: {', '.join(r['matched_skills'])}")

    print(f"\n{len(results)} total matches stored.")
    if stretch and not args.show_stretch:
        print(f"{len(stretch)} hidden as stretch roles (more than {gate:g} years "
              f"short). Re-run with --show-stretch to see them.")
    elif stretch:
        print(f"{len(stretch)} of these are stretch roles, marked [STRETCH].")
    conn.close()


def _seniority_note(verdict) -> str:
    """One-line provenance for a seniority score: requirement, gap, and origin."""
    if verdict.required_years is None:
        return "years: no requirement stated (source=none)"
    gap = "candidate years unset" if verdict.gap is None else f"gap {verdict.gap:+g}y"
    return (f"needs {verdict.required_years:g}y, {gap} "
            f"(source={verdict.source})")


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


# The evaluation set names job ids, not postings. Deleting a row it points at
# turns a scored entry into a dangling reference and silently changes what the
# eval measures, so prune treats these ids as protected.
GOLD_SET_PATH = Path(__file__).resolve().parents[2] / "tests" / "eval" / "skills_gold.yaml"


def gold_referenced_job_ids(path: Path | None = None) -> set[int]:
    """Job ids the gold set refers to, which prune must never delete.

    A missing file means there is no gold set to protect and is not an error.
    A file that exists but cannot be read is different: it may well name ids we
    are about to delete, and we cannot tell which, so that raises rather than
    quietly protecting nothing.
    """
    import yaml

    path = path or GOLD_SET_PATH
    if not path.exists():
        logger.debug("no gold set at %s; nothing to protect", path)
        return set()

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"prune: {path} exists but does not parse ({exc.__class__.__name__}). "
            f"It may name rows about to be deleted and we cannot tell which. "
            f"Fix the file, or move it aside, before pruning."
        ) from exc

    entries = data.get("entries", data) if isinstance(data, dict) else data
    ids = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        job_id = entry.get("job_id")
        if isinstance(job_id, int):
            ids.add(job_id)
        elif job_id is not None:
            # An unresolved placeholder protects nothing; say so rather than
            # letting it look like coverage.
            logger.warning(
                "gold set entry has a non-integer job_id %r - it protects no row",
                job_id,
            )
    return ids


def cmd_prune(args: argparse.Namespace) -> None:
    """Remove stored postings that the current language filter would reject.

    The filter runs at ingest, so it only affects new fetches. Rows stored
    before it was configured, or before it was narrowed, stay in the database
    and keep being scored. This reconciles what is stored with what the config
    now says to keep. Dry run unless --apply is passed.
    """
    from collections import Counter

    config = load_config(Path(args.config))
    conn, db_path = _setup_db(config)

    # Only sources that actually declare a filter are candidates. A source with
    # no `languages` key has not opted in and is left alone.
    filters = {
        src["adapter"]: {lang.lower() for lang in src["languages"]}
        for src in config.get("sources", [])
        if src.get("languages")
    }
    if not filters:
        print("No source declares a `languages` filter. Nothing to prune.")
        conn.close()
        return

    print(f"Database: {db_path}")
    print("Language filters in effect:")
    for adapter, langs in sorted(filters.items()):
        print(f"  {adapter}: keep {', '.join(sorted(langs))}")

    doomed: list[int] = []
    by_source_lang: Counter = Counter()
    for adapter, langs in filters.items():
        rows = conn.execute(
            "SELECT id, language FROM jobs WHERE source = ?", (adapter,),
        ).fetchall()
        for row in rows:
            language = (row["language"] or "").lower()[:2]
            if not language:
                # No detected language is not a failed filter - we cannot say
                # it fails, so it is kept and reported rather than deleted.
                by_source_lang[(adapter, "(undetected, kept)")] += 1
                continue
            if language not in langs:
                doomed.append(row["id"])
                by_source_lang[(adapter, language)] += 1

    protected = gold_referenced_job_ids()
    skipped = sorted(set(doomed) & protected)
    if skipped:
        doomed = [job_id for job_id in doomed if job_id not in protected]
        print(f"\nProtected by the gold set, not deleted: {len(skipped)}")
        for job_id in skipped:
            row = conn.execute(
                "SELECT source, language, title FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
            print(f"      {job_id:>6}  {row['source']:<6} {row['language'] or '?':<3}"
                  f"  {(row['title'] or '')[:52]}")
            logger.info(
                "prune: skipping job %s (%s/%s) - referenced by %s",
                job_id, row["source"], row["language"], GOLD_SET_PATH.name,
            )
        # The per-language tally above counted them, so correct it.
        for job_id in skipped:
            row = conn.execute(
                "SELECT source, language FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
            by_source_lang[(row["source"], (row["language"] or "").lower()[:2])] -= 1

    total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    print(f"\nStored jobs: {total_jobs}")
    print(f"Failing the filter: {len(doomed)}")
    print("\n  by source and language:")
    for (adapter, language), count in sorted(
        by_source_lang.items(), key=lambda kv: (-kv[1], kv[0]),
    ):
        if count == 0:
            # Zeroed out by the gold-set exclusion above; the rows are already
            # listed there by id, so a 0 line here is noise.
            continue
        print(f"      {adapter:<8} {language:<22} {count:>5}")

    if not doomed:
        print("\nNothing to prune.")
        conn.close()
        return

    placeholders = ",".join("?" * len(doomed))
    matches = conn.execute(
        f"SELECT COUNT(*) FROM matches WHERE job_id IN ({placeholders})",  # noqa: S608
        doomed,
    ).fetchone()[0]
    saved = conn.execute(
        f"SELECT COUNT(*) FROM matches WHERE job_id IN ({placeholders})"   # noqa: S608
        " AND status IS NOT NULL AND status <> 'dismissed'",
        doomed,
    ).fetchone()[0]
    print(f"\nScored matches that would be removed: {matches}")
    if saved:
        print(f"  WARNING: {saved} of them carry a status you set by hand "
              f"(saved/applied). Pruning discards that.")

    if not args.apply:
        print("\nDRY RUN - nothing was deleted. Re-run with --apply to execute.")
        conn.close()
        return

    # matches.job_id declares ON DELETE CASCADE, but foreign keys are only
    # enforced when the pragma is on for this connection, so the dependent rows
    # are removed explicitly rather than on that assumption.
    conn.execute(
        f"DELETE FROM matches WHERE job_id IN ({placeholders})", doomed,  # noqa: S608
    )
    deleted = conn.execute(
        f"DELETE FROM jobs WHERE id IN ({placeholders})", doomed,         # noqa: S608
    ).rowcount
    conn.commit()
    print(f"\nDeleted {deleted} job(s) and {matches} match(es).")
    print(f"Jobs remaining: {conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]}")
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
    match_parser.add_argument(
        "--show-stretch", action="store_true",
        help="Include jobs filtered out for requiring too many years",
    )

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch jobs from configured sources")
    fetch_parser.add_argument("--sources", default=None, help="Comma-separated adapter names (default: all enabled)")
    fetch_parser.add_argument("-v", "--verbose", action="store_true", help="Show debug output")

    # prune
    prune_parser = subparsers.add_parser(
        "prune",
        help="Remove stored postings the current language filter would reject",
    )
    prune_parser.add_argument(
        "--apply", action="store_true",
        help="actually delete (default: dry run, report only)",
    )

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
    elif args.command == "prune":
        cmd_prune(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()