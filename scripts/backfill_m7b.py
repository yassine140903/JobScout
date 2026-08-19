#!/usr/bin/env python
"""Backfill M7b: populate the years columns and rescore.

Re-fetches WTTJ postings so the structured fields the old adapter discarded
(experience_level_minimum, education_level, salary) land on existing rows,
then recomputes matches and reports what the seniority gate now removes:

    uv run python scripts/backfill_m7b.py
    uv run python scripts/backfill_m7b.py --gate 3 --no-fetch

Idempotent: re-running re-fetches the same postings, rewrites the same
columns, and recomputes the same matches. Nothing is inserted twice.

Not part of the package.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from jobscout.config import load_config                                # noqa: E402
from jobscout.db import (                                              # noqa: E402
    get_connection, migrate_m2, migrate_m3, migrate_m4, migrate_m5,
    migrate_m6, migrate_m7b, migrate_m7d,
)
from jobscout.matching import (                                       # noqa: E402
    DEFAULT_FILTER_ON_INFERRED, DEFAULT_GATE_YEARS, run_matching,
)
from jobscout.profiles import resolve_candidate_years                  # noqa: E402
from jobscout.sources import enrich_config_from_profile, normalize     # noqa: E402
from jobscout.sources.wttj import WTTJAdapter                          # noqa: E402

# Columns this backfill owns. Written by source_id, so a re-run is a no-op.
M7B_JOB_COLUMNS = (
    "required_years_min", "education_level",
    "salary_yearly_min", "salary_currency", "seniority_source",
)


def seniority_distribution(conn, profile_name: str) -> Counter:
    """Histogram of the stored seniority multiplier, bucketed for legibility."""
    counts: Counter = Counter()
    rows = conn.execute(
        "SELECT m.facet_scores FROM matches m JOIN profiles p ON p.id = m.profile_id"
        " WHERE p.name = ?",
        (profile_name,),
    ).fetchall()
    for (facets,) in rows:
        try:
            value = json.loads(facets or "{}").get("seniority")
        except json.JSONDecodeError:
            value = None
        counts[value if value is None else round(float(value), 2)] += 1
    return counts


def print_distribution(title: str, counts: Counter) -> None:
    total = sum(counts.values()) or 1
    print(f"\n  {title}  (n={total})")
    if not counts:
        print("      (no matches)")
        return
    for value, count in sorted(counts.items(), key=lambda kv: (kv[0] is None, -kv[1])):
        bar = "#" * max(1, round(40 * count / total))
        label = "null" if value is None else f"{value:.2f}"
        print(f"      {label:>6}  {count:>5}  {bar}")


def refetch_wttj(conn, config: dict) -> tuple[int, int]:
    """Re-fetch WTTJ and write the M7b columns onto existing rows.

    Returns (postings_seen, rows_updated). Matches on (source, source_id), the
    same key the insert path dedups on, so nothing is duplicated.
    """
    source_cfg = next(
        (s for s in config.get("sources", []) if s.get("adapter") == "wttj"),
        {"keywords": []},
    )
    if not source_cfg.get("keywords"):
        print("  no WTTJ keywords configured (profile has no skills/domains?) — "
              "skipping re-fetch")
        return 0, 0

    postings = WTTJAdapter().fetch(source_cfg)
    updated = 0
    assignments = ", ".join(f"{c} = :{c}" for c in M7B_JOB_COLUMNS)

    for posting in postings:
        row = normalize(posting)
        params = {c: row[c] for c in M7B_JOB_COLUMNS}
        params["source_id"] = row["source_id"]
        cur = conn.execute(
            f"UPDATE jobs SET {assignments} "  # noqa: S608 — fixed column names
            "WHERE source = 'wttj' AND source_id = :source_id",
            params,
        )
        updated += cur.rowcount
    conn.commit()
    return len(postings), updated


def clear_stale_seniority(conn) -> int:
    """Remove the numbers-as-strings the deleted _map_seniority left behind.

    Those values ("3", "0.5") were never valid seniority buckets — they are the
    bug's residue, and they now live properly in required_years_min. Buckets
    from other adapters (eures, euraxess) are left alone.
    """
    cur = conn.execute(
        "UPDATE jobs SET seniority = NULL "
        "WHERE source = 'wttj' AND seniority GLOB '[0-9]*'"
    )
    conn.commit()
    return cur.rowcount


def coverage(conn) -> None:
    """How much of the corpus now carries a stated requirement."""
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    with_years = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE required_years_min IS NOT NULL"
    ).fetchone()[0]
    print(f"\n  jobs with a stated years requirement: {with_years} / {total}")
    print("  by source:")
    for row in conn.execute(
        "SELECT source, COUNT(*) AS n,"
        "       SUM(required_years_min IS NOT NULL) AS with_years,"
        "       SUM(education_level IS NOT NULL) AS with_edu,"
        "       SUM(salary_yearly_min IS NOT NULL) AS with_salary"
        " FROM jobs GROUP BY source ORDER BY n DESC"
    ):
        print(f"      {row['source']:<10} {row['n']:>5} jobs  "
              f"years={row['with_years']:<5} edu={row['with_edu']:<5} "
              f"salary={row['with_salary']}")

    print("  distinct required_years_min values:")
    for row in conn.execute(
        "SELECT required_years_min AS y, COUNT(*) AS n FROM jobs"
        " WHERE required_years_min IS NOT NULL GROUP BY y ORDER BY y"
    ):
        print(f"      {row['y']:>5g} years  {row['n']:>5}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="default", help="profile to rescore")
    parser.add_argument("--gate", type=float, default=None,
                        help=f"years gate (default: config, else {DEFAULT_GATE_YEARS})")
    parser.add_argument("--no-fetch", action="store_true",
                        help="skip the network re-fetch, only rescore")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    config = load_config(Path(args.config))
    conn = get_connection(Path(config["db_path"]))
    for migrate in (migrate_m2, migrate_m3, migrate_m4, migrate_m5,
                    migrate_m6, migrate_m7b, migrate_m7d):
        migrate(conn)

    profile = conn.execute(
        "SELECT * FROM profiles WHERE name = ?", (args.profile,)
    ).fetchone()
    if profile is None:
        print(f"ERROR: profile {args.profile!r} not found. Run 'jobscout ingest' first.",
              file=sys.stderr)
        return 1

    scoring = config.get("scoring", {})
    seniority_cfg = scoring.get("seniority") or {}
    gate = args.gate if args.gate is not None else seniority_cfg.get(
        "gate_years", DEFAULT_GATE_YEARS,
    )
    filter_on_inferred = seniority_cfg.get(
        "filter_on_inferred", DEFAULT_FILTER_ON_INFERRED,
    )
    candidate_years, years_source = resolve_candidate_years(profile, config)

    print("M7b backfill")
    print(f"  profile        : {args.profile}")
    print(f"  candidate years: "
          f"{'unset' if candidate_years is None else f'{candidate_years:g}'} "
          f"({years_source})")
    print(f"  gate           : {gate:g} years")
    print(f"  filter on inferred: {'yes' if filter_on_inferred else 'no'}")

    before = seniority_distribution(conn, args.profile)
    print_distribution("seniority multiplier BEFORE", before)

    # --- Re-fetch ---
    if args.no_fetch:
        print("\n  --no-fetch: skipping the WTTJ re-fetch")
    else:
        print("\n  re-fetching WTTJ ...")
        enriched = enrich_config_from_profile(config, profile)
        try:
            seen, updated = refetch_wttj(conn, enriched)
            print(f"  {seen} postings fetched, {updated} existing rows updated")
        except Exception as exc:
            print(f"  WARNING: re-fetch failed ({type(exc).__name__}: {exc}); "
                  f"continuing with what is already stored", file=sys.stderr)

    cleared = clear_stale_seniority(conn)
    print(f"  cleared {cleared} stale numeric jobs.seniority value(s) "
          f"left by the removed enum mapping")

    coverage(conn)

    # --- Rescore ---
    print("\n  rescoring ...")
    results = run_matching(
        conn, args.profile,
        weights=scoring.get("weights"),
        candidate_years=candidate_years,
        gate_years=gate,
        filter_on_inferred=filter_on_inferred,
    )

    after = seniority_distribution(conn, args.profile)
    print_distribution("seniority multiplier AFTER", after)

    # --- Gate impact ---
    filtered = [r for r in results if r["filtered"]]
    by_source: Counter = Counter(r["seniority"].source for r in filtered)
    print(f"\n  filtered at gate {gate:g}: {len(filtered)} of {len(results)} matches "
          f"({100.0 * len(filtered) / max(1, len(results)):.1f}%)")
    for source, count in by_source.most_common():
        print(f"      requirement source={source}: {count}")

    neutral_before = before.get(1.0, 0)
    neutral_after = after.get(1.0, 0)
    print(f"\n  scoring a flat 1.0: {neutral_before} -> {neutral_after}")

    if filtered:
        print("\n  most-filtered examples:")
        for r in sorted(filtered, key=lambda r: -r["seniority"].gap)[:8]:
            sen = r["seniority"]
            print(f"      needs {sen.required_years:g}y (gap {sen.gap:+g}, "
                  f"{sen.source}): {r['job_title'][:56]}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
