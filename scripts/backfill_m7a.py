#!/usr/bin/env python
"""M7a backfill: clean stored descriptions, re-extract the CV, re-embed, re-match.

    uv run python scripts/backfill_m7a.py --dry-run
    uv run python scripts/backfill_m7a.py

Idempotent: clean_description is a fixpoint and CV re-extraction is
deterministic, so a second run reports zero changes and simply recomputes
matches. Not part of the package.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from jobscout.config import load_config  # noqa: E402
from jobscout.profiles import RuleBasedExtractor, extract_text  # noqa: E402
from jobscout.textclean import (  # noqa: E402
    check_text_quality,
    clean_description,
    log_quality_warning,
)

PROFILE_FACET_FIELDS = ("skills", "domains", "languages")


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Step 1 - descriptions
# ---------------------------------------------------------------------------

def clean_descriptions(conn: sqlite3.Connection, dry_run: bool) -> dict:
    banner("STEP 1  Clean stored job descriptions")

    rows = conn.execute(
        "SELECT id, description FROM jobs WHERE description IS NOT NULL"
    ).fetchall()

    changed: list[tuple[int, int, int]] = []  # (job_id, before_len, after_len)
    emptied = 0

    for row in rows:
        before = row["description"]
        after = clean_description(before)
        if after == before:
            continue
        if not after:
            # Cleaning must never silently destroy a description.
            emptied += 1
            print(f"  WARNING: job {row['id']} cleaned to empty text - left unchanged")
            continue
        changed.append((row["id"], len(before), len(after)))
        if not dry_run:
            conn.execute(
                """
                UPDATE jobs
                SET description = ?, skills_embedding = NULL, domain_embedding = NULL
                WHERE id = ?
                """,
                (after, row["id"]),
            )

    if not dry_run:
        conn.commit()

    print(f"  descriptions inspected : {len(rows)}")
    print(f"  descriptions changed   : {len(changed)}")
    if emptied:
        print(f"  refused (would empty)  : {emptied}")

    stats = {"inspected": len(rows), "changed": len(changed), "emptied": emptied}
    if changed:
        before_total = sum(b for _, b, _ in changed)
        after_total = sum(a for _, _, a in changed)
        reductions = [b - a for _, b, a in changed]
        stats.update(
            {
                "mean_before": before_total / len(changed),
                "mean_after": after_total / len(changed),
                "mean_reduction": statistics.mean(reductions),
                "pct_reduction": 100 * (before_total - after_total) / before_total,
            }
        )
        print(f"  mean length before     : {stats['mean_before']:.0f} chars")
        print(f"  mean length after      : {stats['mean_after']:.0f} chars")
        print(
            f"  mean reduction         : {stats['mean_reduction']:.0f} chars "
            f"({stats['pct_reduction']:.1f}%)"
        )
        biggest = max(changed, key=lambda c: c[1] - c[2])
        print(
            f"  largest single change  : job {biggest[0]} "
            f"{biggest[1]} -> {biggest[2]} chars"
        )
    else:
        print("  nothing to change - descriptions are already clean")

    return stats


# ---------------------------------------------------------------------------
# Step 2 - CV
# ---------------------------------------------------------------------------

def reextract_cv(
    conn: sqlite3.Connection, cv_path: Path, x_tolerance: float | None, dry_run: bool
) -> dict:
    banner("STEP 2  Re-extract the CV")

    profile = conn.execute("SELECT * FROM profiles ORDER BY id LIMIT 1").fetchone()
    if profile is None:
        print("  no profile stored - skipped")
        return {"status": "no profile"}

    if not cv_path.exists():
        print(f"  WARNING: {cv_path} not found - profiles.raw_text left untouched.")
        print("  Re-run with --cv PATH once the source PDF is available.")
        return {"status": "cv missing"}

    old_text = profile["raw_text"] or ""
    new_text = extract_text(cv_path, x_tolerance=x_tolerance)

    old_report = check_text_quality(old_text) if old_text else None
    new_report = check_text_quality(new_text)

    print(f"  profile                : '{profile['name']}' (id {profile['id']})")
    print(f"  source                 : {cv_path}")
    if old_report:
        print(f"  before                 : {len(old_text)} chars | {old_report.summary()}")
        print(f"                           ok={old_report.ok} {old_report.reasons}")
    print(f"  after                  : {len(new_text)} chars | {new_report.summary()}")
    print(f"                           ok={new_report.ok} {new_report.reasons}")

    if not new_report.ok:
        # Extraction already retried its fallbacks; surface what still looks wrong.
        log_quality_warning(new_report, str(cv_path))
        print("  WARNING: re-extracted CV still fails the quality guard (see log).")

    if new_text == old_text:
        print("  raw_text unchanged")
        return {"status": "unchanged", "chars": len(new_text)}

    facets = RuleBasedExtractor().extract(new_text)
    print(f"  skills                 : {len(facets['skills'])} detected")
    print(f"  domains                : {len(facets['domains'])} detected")
    print(f"  seniority              : {facets['seniority']}")

    if not dry_run:
        conn.execute(
            """
            UPDATE profiles
            SET raw_text = ?, skills = ?, domains = ?, seniority = ?, languages = ?,
                skills_embedding = NULL, domain_embedding = NULL,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (
                new_text,
                json.dumps(facets["skills"]),
                json.dumps(facets["domains"]),
                facets["seniority"],
                json.dumps(facets["languages"]),
                profile["id"],
            ),
        )
        conn.commit()

    return {
        "status": "updated",
        "chars_before": len(old_text),
        "chars_after": len(new_text),
        "tokens_before": old_report.n_tokens if old_report else None,
        "tokens_after": new_report.n_tokens,
        "skills": len(facets["skills"]),
    }


# ---------------------------------------------------------------------------
# Step 3 - re-embed and re-match
# ---------------------------------------------------------------------------

def rematch(conn: sqlite3.Connection, config: dict, dry_run: bool) -> dict:
    banner("STEP 3  Re-embed and recompute matches")

    profile = conn.execute("SELECT * FROM profiles ORDER BY id LIMIT 1").fetchone()
    if profile is None:
        print("  no profile stored - skipped")
        return {"status": "no profile"}

    pending = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE skills_embedding IS NULL OR domain_embedding IS NULL"
    ).fetchone()[0]
    print(f"  jobs needing embeddings: {pending}")

    if dry_run:
        print("  (dry run - skipping embedding and matching)")
        return {"status": "dry run", "pending": pending}

    from jobscout.embedder import Embedder
    from jobscout.matching import run_matching

    print(f"  loading embedder '{config['model']['name']}' ...")
    embedder = Embedder(config["model"]["name"])

    started = time.perf_counter()
    results = run_matching(
        conn,
        profile["name"],
        embedder=embedder,
        weights=config.get("scoring", {}).get("weights"),
    )
    elapsed = time.perf_counter() - started

    print(f"  matches recomputed     : {len(results)} in {elapsed:.1f}s")
    if results:
        print(f"  top score              : {results[0]['final_score']:.4f}")
        print(f"  bottom score           : {results[-1]['final_score']:.4f}")

    return {"status": "ok", "matches": len(results), "seconds": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--cv", default=str(REPO_ROOT / "cv.pdf"))
    parser.add_argument(
        "--dry-run", action="store_true", help="report changes without writing"
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    db_path = Path(config["db_path"])
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        return 1

    x_tolerance = config.get("extraction", {}).get("pdf_x_tolerance")

    banner("M7a BACKFILL" + ("  (DRY RUN)" if args.dry_run else ""))
    print(f"  database    : {db_path}")
    print(f"  cv          : {args.cv}")
    print(f"  x_tolerance : {x_tolerance}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        desc_stats = clean_descriptions(conn, args.dry_run)
        cv_stats = reextract_cv(conn, Path(args.cv), x_tolerance, args.dry_run)
        match_stats = rematch(conn, config, args.dry_run)
    finally:
        conn.close()

    banner("SUMMARY" + ("  (DRY RUN - nothing written)" if args.dry_run else ""))
    print(
        f"  descriptions : {desc_stats['changed']} of {desc_stats['inspected']} changed"
        + (
            f", mean -{desc_stats['mean_reduction']:.0f} chars "
            f"({desc_stats['pct_reduction']:.1f}%)"
            if desc_stats["changed"]
            else ""
        )
    )
    if cv_stats["status"] == "updated":
        print(
            f"  cv           : rewritten, {cv_stats['chars_before']} -> "
            f"{cv_stats['chars_after']} chars, "
            f"{cv_stats['tokens_before']} -> {cv_stats['tokens_after']} tokens"
        )
    else:
        print(f"  cv           : {cv_stats['status']}")
    if match_stats["status"] == "ok":
        print(f"  matches      : {match_stats['matches']} recomputed")
    else:
        print(f"  matches      : {match_stats['status']}")

    if not args.dry_run:
        print(
            "\n  Next: uv run python scripts/score_distribution.py "
            "--out scripts/out/after_scores.md "
            "--compare scripts/out/baseline_scores.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
