#!/usr/bin/env python
"""Evaluate a skill extractor against the hand-labelled gold set.

    uv run python tests/eval/run_eval.py
    uv run python tests/eval/run_eval.py --extractor rule_based --verbose

Reports micro/macro precision, recall and F1 on skills, plus exact-match
accuracy on required_years_min. Entries whose `expected_skills` is still empty
are counted as unlabelled and reported separately - the whole point of the
gold file is that a human fills it in, so an unlabelled file scores 0.0 rather
than pretending to pass.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

GOLD_PATH = Path(__file__).resolve().parent / "skills_gold.yaml"


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _rule_based_skills(text: str) -> list[str]:
    from jobscout.profiles import RuleBasedExtractor

    return RuleBasedExtractor().extract_from_text(text)["skills"]


EXTRACTORS = {
    "rule_based": _rule_based_skills,
}


def predict_required_years_min(text: str, lang_hint: str | None = None) -> float | None:
    """What the description layer reads out of the posting, or None.

    This is the M7c parser the matching pipeline actually uses, guards and
    all. It replaced a bare regex over the CV extractor's YEARS_PATTERN, which
    matched any number followed by a time unit and so scored a baseline the
    product never ran.
    """
    from jobscout.profiles import parse_required_years

    return parse_required_years(text, lang_hint)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_skill(skill: str) -> str:
    return " ".join(str(skill).lower().split())


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Gold set
# ---------------------------------------------------------------------------

def load_gold(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"ERROR: gold file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not data:
        return []
    entries = data.get("entries", data) if isinstance(data, dict) else data
    return entries or []


def resolve_text(entry: dict, conn: sqlite3.Connection | None) -> str:
    """Full stored description when the DB is reachable, else the excerpt.

    Scoring against the excerpt alone would understate recall, so the DB is
    preferred and the fallback is reported.
    """
    if conn is not None and entry.get("job_id") is not None:
        row = conn.execute(
            "SELECT title, description FROM jobs WHERE id = ?", (entry["job_id"],)
        ).fetchone()
        if row and row["description"]:
            return f"{row['title'] or ''}\n{row['description']}"
    return f"{entry.get('title') or ''}\n{entry.get('description_excerpt') or ''}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(GOLD_PATH))
    parser.add_argument("--db", default=str(REPO_ROOT / "jobscout.db"))
    parser.add_argument(
        "--extractor",
        default="rule_based",
        choices=sorted(EXTRACTORS),
        help="which extractor to evaluate",
    )
    parser.add_argument("--verbose", action="store_true", help="per-entry breakdown")
    args = parser.parse_args()

    entries = load_gold(Path(args.gold))
    extract = EXTRACTORS[args.extractor]

    db_path = Path(args.db)
    conn = None
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

    print("=" * 70)
    print(f"Skill extraction eval - extractor: {args.extractor}")
    print("=" * 70)
    print(f"  gold file : {Path(args.gold).name}")
    print(f"  entries   : {len(entries)}")
    print(f"  text from : {'stored descriptions' if conn else 'excerpts (DB not found)'}")

    if not entries:
        print("\n  Gold file has no entries. Nothing to score.")
        print("\n  skills   precision=0.0000 recall=0.0000 f1=0.0000")
        print("  years    accuracy=0.0000")
        return 0

    total_tp = total_fp = total_fn = 0
    per_entry_f1: list[float] = []
    labelled = 0
    years_labelled = 0
    years_correct = 0
    years_misses: list[tuple] = []
    rows_out: list[tuple] = []

    try:
        for entry in entries:
            expected_raw = entry.get("expected_skills") or []
            expected = {normalize_skill(s) for s in expected_raw}
            text = resolve_text(entry, conn)
            predicted = {normalize_skill(s) for s in extract(text)}

            if expected:
                labelled += 1
                tp = len(expected & predicted)
                fp = len(predicted - expected)
                fn = len(expected - predicted)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                per_entry_f1.append(prf(tp, fp, fn)[2])
            else:
                tp = fp = fn = 0

            # A years label of null is an assertion - "this posting states no
            # requirement, the parser must return None" - and is only
            # distinguishable from "not yet labelled" by whether the entry has
            # been labelled at all. expected_skills is the marker for that.
            if expected:
                expected_years = entry.get("expected_required_years_min")
                years_labelled += 1
                predicted_years = predict_required_years_min(
                    text, entry.get("language"),
                )
                if expected_years is None:
                    correct = predicted_years is None
                else:
                    correct = (
                        predicted_years is not None
                        and float(predicted_years) == float(expected_years)
                    )
                years_correct += correct
                if not correct:
                    # A miss where the structured field carries the number is
                    # not a parser failure: the posting never wrote it in prose.
                    # Without this the metric conflates the two.
                    api_years = None
                    if conn is not None and entry.get("job_id") is not None:
                        row = conn.execute(
                            "SELECT required_years_min FROM jobs WHERE id = ?",
                            (entry["job_id"],),
                        ).fetchone()
                        api_years = row["required_years_min"] if row else None
                    years_misses.append(
                        (entry.get("job_id"), expected_years, predicted_years,
                         api_years),
                    )

            rows_out.append(
                (
                    entry.get("job_id"),
                    (entry.get("title") or "")[:40],
                    len(expected),
                    len(predicted),
                    tp,
                    fp,
                    fn,
                )
            )
    finally:
        if conn is not None:
            conn.close()

    unlabelled = len(entries) - labelled
    precision, recall, f1 = prf(total_tp, total_fp, total_fn)
    macro_f1 = sum(per_entry_f1) / len(per_entry_f1) if per_entry_f1 else 0.0
    years_accuracy = years_correct / years_labelled if years_labelled else 0.0

    if args.verbose:
        print("\n  per entry:")
        print(f"    {'job':>6} {'title':<40} {'gold':>4} {'pred':>4} {'tp':>3} {'fp':>3} {'fn':>3}")
        for job_id, title, n_exp, n_pred, tp, fp, fn in rows_out:
            print(f"    {str(job_id):>6} {title:<40} {n_exp:>4} {n_pred:>4} {tp:>3} {fp:>3} {fn:>3}")

    print(f"\n  labelled entries   : {labelled}/{len(entries)}")
    if unlabelled:
        print(f"  UNLABELLED         : {unlabelled} (expected_skills still empty)")

    print("\n  skills (micro)")
    print(f"    precision : {precision:.4f}")
    print(f"    recall    : {recall:.4f}")
    print(f"    f1        : {f1:.4f}")
    print(f"    tp={total_tp} fp={total_fp} fn={total_fn}")
    print(f"\n  skills (macro f1)   : {macro_f1:.4f}")
    print(f"\n  required_years_min  (parse_required_years, M7c)")
    print(f"    labelled  : {years_labelled}/{len(entries)}")
    print(f"    accuracy  : {years_accuracy:.4f} ({years_correct}/{years_labelled or 0})")
    if years_misses:
        print("    misses    :")
        for job_id, want, got, api_years in years_misses:
            note = ""
            if got is None and api_years is not None:
                note = (f" - not in the prose; the source's structured field "
                        f"says {api_years:g}, so the chain still resolves it")
            print(f"      job {job_id}: expected {want}, predicted {got}{note}")

    if unlabelled == len(entries):
        print(
            "\n  All entries are unlabelled - fill in expected_skills and "
            "expected_required_years_min in the gold file to get a real score."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
