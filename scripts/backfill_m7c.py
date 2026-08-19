#!/usr/bin/env python
"""Backfill M7c: read experience requirements out of stored descriptions.

The structured field is null for most of the corpus, and often null for
postings that state a requirement in plain prose. This runs the description
parser over every job whose ``required_years_min`` is null, reports what it
found, and recomputes matches so the new layer takes effect:

    uv run python scripts/backfill_m7c.py
    uv run python scripts/backfill_m7c.py --sample 40 --gate 3

Idempotent, and by construction: the parsed figure is not written back to
``jobs.required_years_min``. That column stays what M7b made it - the source's
own structured field - and the description layer resolves at match time inside
``resolve_seniority``. Re-running re-reads the same descriptions and recomputes
the same matches. Nothing is inserted twice.

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
from jobscout.matching import (                                        # noqa: E402
    DEFAULT_FILTER_ON_INFERRED, DEFAULT_GATE_YEARS, run_matching,
)
from jobscout.profiles import (                                        # noqa: E402
    find_required_years, resolve_candidate_years,
)

SNIPPET_SAMPLE = 20


def scan_descriptions(conn) -> tuple[list[dict], int]:
    """Parse every description whose structured field said nothing.

    Returns (hits, candidates_scanned). A hit carries the job, the years and
    the snippet that produced them, so the regex can be eyeballed rather than
    trusted.
    """
    rows = conn.execute(
        "SELECT id, title, company, language, description FROM jobs"
        " WHERE required_years_min IS NULL"
        "   AND description IS NOT NULL AND TRIM(description) <> ''"
        " ORDER BY id"
    ).fetchall()

    hits: list[dict] = []
    for row in rows:
        found = find_required_years(row["description"], row["language"])
        if found is None:
            continue
        hits.append({
            "job_id": row["id"],
            "title": row["title"] or "",
            "language": row["language"] or "?",
            "years": found.years,
            "matched": found.matched,
            "snippet": found.snippet,
            "pattern_language": found.language,
        })
    return hits, len(rows)


def evenly_spaced(items: list, k: int) -> list:
    """A deterministic spread of k items, so the sample is not all one source."""
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def filtered_job_ids(conn, profile_name: str) -> dict[int, str]:
    """job_id -> requirement source, for every match the gate currently hides."""
    hidden: dict[int, str] = {}
    rows = conn.execute(
        "SELECT m.job_id, m.facet_scores FROM matches m"
        " JOIN profiles p ON p.id = m.profile_id WHERE p.name = ?",
        (profile_name,),
    ).fetchall()
    for job_id, facets in rows:
        try:
            detail = json.loads(facets or "{}").get("_seniority") or {}
        except json.JSONDecodeError:
            continue
        if detail.get("filtered"):
            hidden[job_id] = detail.get("source") or "unknown"
    return hidden


def print_distribution(title: str, counts: Counter, fmt=str) -> None:
    total = sum(counts.values()) or 1
    print(f"\n  {title}  (n={total})")
    if not counts:
        print("      (nothing)")
        return
    for value, count in sorted(counts.items()):
        bar = "#" * max(1, round(40 * count / total))
        print(f"      {fmt(value):>6}  {count:>5}  {bar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="default", help="profile to rescore")
    parser.add_argument("--gate", type=float, default=None,
                        help=f"years gate (default: config, else {DEFAULT_GATE_YEARS})")
    parser.add_argument("--sample", type=int, default=SNIPPET_SAMPLE,
                        help="matched snippets to print for manual review")
    parser.add_argument("--no-rescore", action="store_true",
                        help="report what the parser finds, change nothing")
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

    total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    with_api = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE required_years_min IS NOT NULL"
    ).fetchone()[0]

    print("M7c backfill - experience requirements from description prose")
    print(f"  profile        : {args.profile}")
    print(f"  candidate years: "
          f"{'unset' if candidate_years is None else f'{candidate_years:g}'} "
          f"({years_source})")
    print(f"  gate           : {gate:g} years")
    print(f"  corpus         : {total_jobs} jobs, {with_api} with a structured figure")

    # --- Layer 2: what the prose says -------------------------------------
    hits, scanned = scan_descriptions(conn)
    print(f"\n  scanned {scanned} description(s) with no structured figure")
    print(f"  gained a years figure from the description layer: {len(hits)} "
          f"({100.0 * len(hits) / max(1, scanned):.1f}% of them, "
          f"{100.0 * len(hits) / max(1, total_jobs):.1f}% of the corpus)")

    print_distribution(
        "extracted values", Counter(h["years"] for h in hits), lambda y: f"{y:g}y",
    )
    print_distribution(
        "by posting language", Counter(h["language"] for h in hits),
    )

    if args.no_rescore:
        print("\n  --no-rescore: stopping before the rescore")
        _print_snippets(hits, args.sample)
        conn.close()
        return 0

    # --- Rescore ----------------------------------------------------------
    before = filtered_job_ids(conn, args.profile)
    print(f"\n  rescoring ({len(before)} match(es) currently hidden as stretch) ...")
    results = run_matching(
        conn, args.profile,
        weights=scoring.get("weights"),
        candidate_years=candidate_years,
        gate_years=gate,
        filter_on_inferred=filter_on_inferred,
    )
    after = filtered_job_ids(conn, args.profile)

    by_source = Counter(r["seniority"].source for r in results)
    print(f"\n  requirement source across {len(results)} scored match(es):")
    for source, count in by_source.most_common():
        print(f"      {source:<12} {count:>5}  "
              f"({100.0 * count / max(1, len(results)):.1f}%)")

    newly = {job_id: src for job_id, src in after.items() if job_id not in before}
    gone = [job_id for job_id in before if job_id not in after]
    print(f"\n  filtered at gate {gate:g}: {len(after)} of {len(results)} "
          f"(was {len(before)})")
    print(f"      newly filtered: {len(newly)}  "
          f"({sum(1 for s in newly.values() if s == 'description')} of them because "
          f"the description stated a requirement)")
    if gone:
        print(f"      no longer filtered: {len(gone)}")

    described = [r for r in results if r["seniority"].source == "description"]
    if described:
        worst = sorted(described, key=lambda r: -(r["seniority"].gap or 0))[:8]
        print("\n  biggest gaps found in prose:")
        for r in worst:
            sen = r["seniority"]
            flag = " [filtered]" if r["filtered"] else ""
            print(f"      needs {sen.required_years:g}y (gap {sen.gap:+g}){flag}: "
                  f"{(r['job_title'] or '')[:52]}")

    _print_snippets(hits, args.sample)
    conn.close()
    return 0


def _print_snippets(hits: list[dict], k: int) -> None:
    """The point of the sample: see the regex's work before trusting it."""
    sample = evenly_spaced(hits, k)
    if not sample:
        return
    print(f"\n  {len(sample)} matched snippet(s) for manual review "
          f"(spread evenly over {len(hits)} hits):")
    for h in sample:
        print(f"\n    [{h['job_id']}] {h['years']:g}y  ({h['language']}, matched by "
              f"{h['pattern_language']} patterns, on {h['matched']!r})")
        print(f"        {(h['title'] or '')[:70]}")
        print(f"        {h['snippet']}")


if __name__ == "__main__":
    sys.exit(main())
