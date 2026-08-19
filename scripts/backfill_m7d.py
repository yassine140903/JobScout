#!/usr/bin/env python
"""Backfill M7d: replace WTTJ blurbs with the postings' real descriptions.

The Algolia index carries no job description for any posting - only `profile`,
a ~1KB requirements blurb, null on 15% of hits. This fetches every stored WTTJ
job's detail page, rewrites the description, drops the stale embeddings so the
next match run re-embeds the new text, and rescores:

    uv run python scripts/backfill_m7d.py
    uv run python scripts/backfill_m7d.py --limit 50 --dry-run
    uv run python scripts/backfill_m7d.py --concurrency 4

Idempotent: re-running re-fetches the same postings, writes the same text, and
recomputes the same matches. A job whose fetch fails keeps whatever text it
already had - the backfill can never empty a row.

Not part of the package.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
    RuleBasedExtractor, find_required_years, resolve_candidate_years,
)
from jobscout.sources import RawPosting                                # noqa: E402
from jobscout.sources.wttj import (                                    # noqa: E402
    DEFAULT_DETAIL_CONCURRENCY, WTTJAdapter,
)
from jobscout.textclean import clean_description                       # noqa: E402

LONG_DESCRIPTION = 2000   # ~512 e5 tokens: past this a single embedding truncates


def load_wttj_jobs(conn, limit: int | None) -> list:
    """Stored WTTJ rows, newest first, with the raw hit the fetch needs."""
    sql = (
        "SELECT id, title, description, description_source, skills, raw_data"
        " FROM jobs WHERE source = 'wttj' AND raw_data IS NOT NULL ORDER BY id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def to_posting(row) -> RawPosting | None:
    """Rebuild the minimum RawPosting the detail fetch needs from a stored row."""
    try:
        hit = json.loads(row["raw_data"])
    except (json.JSONDecodeError, TypeError):
        return None
    return RawPosting(
        title=row["title"] or "",
        source="wttj",
        source_id=str(row["id"]),
        description=row["description"],
        description_source=row["description_source"] or (
            "blurb" if row["description"] else "none"
        ),
        raw_data=hit,
    )


def describe_lengths(lengths: list[int], label: str) -> None:
    if not lengths:
        print(f"  {label}: (nothing)")
        return
    lengths = sorted(lengths)
    p90 = lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))]
    over = sum(1 for n in lengths if n > LONG_DESCRIPTION)
    print(f"  {label}: n={len(lengths)} min={lengths[0]} median={lengths[len(lengths)//2]} "
          f"mean={statistics.mean(lengths):.0f} p90={p90} max={lengths[-1]}")
    print(f"      over {LONG_DESCRIPTION} chars: {over} ({100.0*over/len(lengths):.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="default", help="profile to rescore")
    parser.add_argument("--limit", type=int, default=None, help="only the first N jobs")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"parallel detail fetches (default: config, else "
                             f"{DEFAULT_DETAIL_CONCURRENCY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and report, write nothing")
    parser.add_argument("--no-rescore", action="store_true",
                        help="write descriptions but skip the match recompute")
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

    wttj_cfg = next(
        (s for s in config.get("sources", []) if s.get("adapter") == "wttj"), {},
    )
    concurrency = args.concurrency or wttj_cfg.get(
        "detail_concurrency", DEFAULT_DETAIL_CONCURRENCY,
    )

    rows = load_wttj_jobs(conn, args.limit)
    print("M7d backfill - real descriptions from the WTTJ detail endpoint")
    print(f"  stored WTTJ jobs : {len(rows)}")
    print(f"  concurrency      : {concurrency}")
    print(f"  mode             : {'DRY RUN, nothing written' if args.dry_run else 'writing'}")

    # --- before ---------------------------------------------------------
    extractor = RuleBasedExtractor()
    before_lengths = [len(r["description"] or "") for r in rows]
    before_null = sum(1 for r in rows if not (r["description"] or "").strip())
    before_empty_skills = sum(
        1 for r in rows if not json.loads(r["skills"] or "[]")
    )
    before_years = sum(
        1 for r in rows
        if r["description"] and find_required_years(r["description"]) is not None
    )
    print("\n  BEFORE")
    describe_lengths([n for n in before_lengths if n], "description length")
    print(f"  rows with no text        : {before_null}")
    print(f"  rows extracting no skills: {before_empty_skills}")
    print(f"  rows yielding a years figure: {before_years}")

    # --- fetch ----------------------------------------------------------
    postings, by_id = [], {}
    for row in rows:
        p = to_posting(row)
        if p is None:
            continue
        postings.append(p)
        by_id[row["id"]] = (row, p)

    print(f"\n  fetching {len(postings)} detail pages ...")
    report = WTTJAdapter().enrich_with_details(postings, concurrency=concurrency)
    print(f"  resolved {report.resolved}/{report.attempted} "
          f"({report.renamed} via a renamed company), "
          f"{report.delisted} delisted, {report.failed} failed "
          f"({100.0 * report.failure_ratio:.1f}%)")
    print("\n  outcomes:")
    for reason, count in sorted((report.reasons or {}).items(), key=lambda kv: -kv[1]):
        print(f"      {reason:<18} {count:>5}")

    # --- write ----------------------------------------------------------
    updated: Counter = Counter()
    after_lengths, after_null, after_empty_skills, after_years = [], 0, 0, 0
    rescued_null, delisted_marked = 0, 0

    for job_id, (row, p) in by_id.items():
        text = clean_description(p.description) or None
        after_lengths.append(len(text or ""))
        updated[p.description_source] += 1

        had_text = bool((row["description"] or "").strip())
        if not had_text and text:
            rescued_null += 1
        if not text:
            after_null += 1
        if text and find_required_years(text) is not None:
            after_years += 1
        # Skills are re-extracted from title + the new text, the same way
        # matching does it, so the before/after comparison is like for like.
        skills = extractor.extract_from_text(f"{row['title'] or ''}\n{text or ''}")["skills"]
        if not skills:
            after_empty_skills += 1
        if p.delisted_at:
            delisted_marked += 1

        if args.dry_run:
            continue
        conn.execute(
            "UPDATE jobs SET description = ?, description_source = ?,"
            "                delisted_at = COALESCE(?, delisted_at),"
            # The stored vectors describe the old text. Clearing them makes the
            # next match run re-embed; leaving them would score new text with
            # old embeddings, silently.
            "                skills = NULL, domains = NULL,"
            "                skills_embedding = NULL, domain_embedding = NULL"
            " WHERE id = ?",
            (text, p.description_source, p.delisted_at, job_id),
        )
    if not args.dry_run:
        conn.commit()

    print("\n  AFTER")
    print("  jobs by description_source:")
    for source, count in updated.most_common():
        print(f"      {source:<8} {count:>5}")
    describe_lengths([n for n in after_lengths if n], "description length")
    print(f"  rows with no text        : {after_null}  (was {before_null})")
    print(f"  rows extracting no skills: {after_empty_skills}  (was {before_empty_skills})")
    print(f"  rows yielding a years figure: {after_years}  (was {before_years})")
    print(f"\n  previously-empty rows that now have text: {rescued_null}")
    print(f"  rows marked delisted                   : {delisted_marked}")

    if args.dry_run:
        print("\n  --dry-run: nothing was written")
        conn.close()
        return 0

    # --- rescore --------------------------------------------------------
    if args.no_rescore:
        print("\n  --no-rescore: descriptions written, matches left stale")
        conn.close()
        return 0

    profile = conn.execute(
        "SELECT * FROM profiles WHERE name = ?", (args.profile,)
    ).fetchone()
    if profile is None:
        print(f"\nERROR: profile {args.profile!r} not found; descriptions were "
              f"written but nothing was rescored.", file=sys.stderr)
        conn.close()
        return 1

    scoring = config.get("scoring", {})
    seniority_cfg = scoring.get("seniority") or {}
    candidate_years, _ = resolve_candidate_years(profile, config)
    print("\n  re-embedding and rescoring (this reruns the model over new text) ...")
    results = run_matching(
        conn, args.profile,
        weights=scoring.get("weights"),
        candidate_years=candidate_years,
        gate_years=seniority_cfg.get("gate_years", DEFAULT_GATE_YEARS),
        filter_on_inferred=seniority_cfg.get(
            "filter_on_inferred", DEFAULT_FILTER_ON_INFERRED,
        ),
    )
    by_source = Counter(r["seniority"].source for r in results)
    print(f"  {len(results)} matches rescored; requirement source:")
    for source, count in by_source.most_common():
        print(f"      {source:<12} {count:>5}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
