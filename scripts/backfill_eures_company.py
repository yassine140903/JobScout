#!/usr/bin/env python
"""Repair EURES company names stored as stringified dicts.

The EURES adapter used to write str(employer_object) into jobs.company, so
rows read "{'name': 'KAISCHOOL', 'legalID': None, ...}" instead of "KAISCHOOL".
The adapter is fixed; this rewrites what is already stored:

    uv run python scripts/backfill_eures_company.py --dry-run
    uv run python scripts/backfill_eures_company.py

Idempotent — an already-clean value is left alone, so re-running changes
nothing. Values that do not parse are reported and left untouched rather
than guessed at.

company feeds dedup_hash, so repaired rows get their hash recomputed too;
otherwise future inserts would dedup against a hash of the dict repr.

Not part of the package.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from jobscout.config import load_config                        # noqa: E402
from jobscout.db import get_connection                         # noqa: E402
from jobscout.sources import compute_dedup_hash                # noqa: E402
from jobscout.sources.eures import repair_company_value        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="eures",
                        help="adapter whose rows to repair (default: eures)")
    parser.add_argument("--all-sources", action="store_true",
                        help="repair every source, not just one")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    config = load_config(Path(args.config))
    conn = get_connection(Path(config["db_path"]))

    if args.all_sources:
        rows = conn.execute("SELECT id, source, title, company FROM jobs").fetchall()
        scope = "all sources"
    else:
        rows = conn.execute(
            "SELECT id, source, title, company FROM jobs WHERE source = ?",
            (args.source,),
        ).fetchall()
        scope = f"source={args.source!r}"

    print(f"EURES company repair — {scope}, {len(rows)} rows"
          f"{' (dry run)' if args.dry_run else ''}")

    outcomes: Counter = Counter()
    examples: list[tuple[str, str]] = []
    unparseable: list[tuple[int, str]] = []

    for row in rows:
        repaired, outcome = repair_company_value(row["company"])
        outcomes[outcome] += 1

        if outcome == "unparseable":
            unparseable.append((row["id"], str(row["company"])[:100]))
            continue
        if outcome != "repaired":
            continue

        if len(examples) < 8:
            examples.append((str(row["company"])[:64], repaired))
        if not args.dry_run:
            conn.execute(
                "UPDATE jobs SET company = ?, dedup_hash = ? WHERE id = ?",
                (repaired, compute_dedup_hash(row["title"], repaired), row["id"]),
            )

    if not args.dry_run:
        conn.commit()

    print(f"\n  repaired    : {outcomes['repaired']}")
    print(f"  already clean: {outcomes['clean']}")
    print(f"  unparseable : {outcomes['unparseable']}  (left untouched)")

    if examples:
        print("\n  examples:")
        for before, after in examples:
            print(f"      {before}...")
            print(f"        -> {after}")

    if unparseable:
        print("\n  unparseable rows:")
        for job_id, value in unparseable[:10]:
            print(f"      #{job_id}  {value}")
        if len(unparseable) > 10:
            print(f"      ... and {len(unparseable) - 10} more")

    # Prove idempotency: nothing should still look like a dict afterwards.
    if not args.dry_run:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE company LIKE '{%'"
        ).fetchone()[0]
        print(f"\n  rows still holding a dict-shaped company: {remaining}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
