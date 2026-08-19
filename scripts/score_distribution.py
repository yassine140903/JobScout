#!/usr/bin/env python
"""Dump the match-score distribution for a profile.

Run before and after a data-hygiene change to make the change falsifiable:

    uv run python scripts/score_distribution.py --out scripts/out/baseline_scores.md
    ... apply fix, backfill ...
    uv run python scripts/score_distribution.py --out scripts/out/after_scores.md \
        --compare scripts/out/baseline_scores.json

Each run writes the markdown report plus a JSON sidecar next to it; --compare
reads a sidecar and prints the side-by-side.

Not part of the package.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT = REPO_ROOT / "scripts" / "out" / "baseline_scores.md"
N_BUCKETS = 10
TOP_N = 20


def percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile - no interpolation, so values stay real observations."""
    if not sorted_vals:
        return 0.0
    rank = int(-(-q * len(sorted_vals) // 1))  # ceil
    return sorted_vals[max(0, min(len(sorted_vals) - 1, rank - 1))]


def company_name(raw: str | None) -> str:
    """Some adapters store the whole org object in `company`; pull the name out."""
    if not raw:
        return "(unknown)"
    raw = raw.strip()
    if raw.startswith("{"):
        import ast

        for parse in (json.loads, ast.literal_eval):
            try:
                obj = parse(raw)
            except (ValueError, SyntaxError):
                continue
            if isinstance(obj, dict) and obj.get("name"):
                return str(obj["name"]).strip()
    return raw


def resolve_profile(conn: sqlite3.Connection, name: str | None) -> sqlite3.Row:
    if name:
        row = conn.execute(
            "SELECT * FROM profiles WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise SystemExit(f"ERROR: no profile named {name!r}")
        return row

    rows = conn.execute("SELECT * FROM profiles ORDER BY id").fetchall()
    if not rows:
        raise SystemExit("ERROR: no profiles in the database.")
    if len(rows) > 1:
        print(
            f"NOTE: {len(rows)} profiles present, using '{rows[0]['name']}'. "
            "Pass --profile to choose another."
        )
    return rows[0]


def seniority_detail(row: sqlite3.Row) -> dict:
    """The stored seniority audit trail for a match, or {} when unreadable.

    This, not the jobs table, is the record of what was actually scored: the
    description layer resolves at match time and never writes back to
    jobs.required_years_min, which stays the source's own structured field.
    """
    try:
        return json.loads(row["facet_scores"] or "{}").get("_seniority") or {}
    except json.JSONDecodeError:
        return {}


def is_stretch(row: sqlite3.Row) -> bool:
    """Whether the seniority gate hid this match from the default view."""
    return bool(seniority_detail(row).get("filtered"))


def collect(conn: sqlite3.Connection, profile_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT m.job_id, m.score, m.facet_scores, j.title, j.company,
               j.required_years_min, j.seniority_source
        FROM matches m
        JOIN jobs j ON j.id = m.job_id
        WHERE m.profile_id = ?
        ORDER BY m.score DESC
        """,
        (profile_id,),
    ).fetchall()


def build_stats(rows: list[sqlite3.Row]) -> dict:
    scores = sorted(r["score"] for r in rows)
    stats = {
        "n": len(scores),
        "min": scores[0],
        "p25": percentile(scores, 0.25),
        "median": percentile(scores, 0.50),
        "p75": percentile(scores, 0.75),
        "max": scores[-1],
        "mean": statistics.mean(scores),
        "stdev": statistics.pstdev(scores),
        "spread": scores[-1] - scores[0],
    }

    # Exact ties: how many rows share a score with at least one other row.
    # A pile of identical scores means the ranking is not actually ranking.
    score_counts = Counter(scores)
    stats["ties"] = sum(n for n in score_counts.values() if n > 1)
    stats["distinct_scores"] = len(score_counts)

    # Seniority: where each requirement came from, and what the gate removed
    # on that authority.
    filtered_by_source: Counter = Counter()
    by_source: Counter = Counter()
    description_rescued = 0        # api said nothing, the description did not
    for row in rows:
        detail = seniority_detail(row)
        if not detail:
            continue
        source = detail.get("source") or "unknown"
        by_source[source] += 1
        if source == "description" and row["required_years_min"] is None:
            description_rescued += 1
        if detail.get("filtered"):
            filtered_by_source[source] += 1
    stats["by_source"] = dict(by_source)
    stats["description_rescued"] = description_rescued
    stats["filtered_by_source"] = dict(filtered_by_source)
    stats["filtered_total"] = sum(filtered_by_source.values())

    # Fixed [0,1] buckets, not [min,max] - so before/after histograms are comparable.
    buckets = [0] * N_BUCKETS
    for score in scores:
        idx = min(N_BUCKETS - 1, max(0, int(score * N_BUCKETS)))
        buckets[idx] += 1
    stats["buckets"] = buckets

    top = rows[:TOP_N]
    stats["top"] = []
    for r in top:
        detail = seniority_detail(r)
        stats["top"].append({
            "job_id": r["job_id"],
            "score": r["score"],
            "title": r["title"],
            "company": company_name(r["company"]),
            # The scored figure, whichever layer produced it; the column only
            # ever holds the api one.
            "required_years_min": detail.get("required_years", r["required_years_min"]),
            "seniority_source": detail.get("source") or r["seniority_source"],
        })
    return stats


def render(stats: dict, profile_name: str) -> str:
    lines = [
        f"# Match score distribution - profile `{profile_name}`",
        "",
        f"{stats['n']} scored matches.",
        "",
        "## Summary",
        "",
        "| statistic | value |",
        "|---|---|",
        f"| min | {stats['min']:.4f} |",
        f"| p25 | {stats['p25']:.4f} |",
        f"| median | {stats['median']:.4f} |",
        f"| p75 | {stats['p75']:.4f} |",
        f"| max | {stats['max']:.4f} |",
        f"| mean | {stats['mean']:.4f} |",
        f"| std dev (population) | {stats['stdev']:.4f} |",
        f"| spread (max - min) | {stats['spread']:.4f} |",
        f"| exact ties | {stats.get('ties', 0)} of {stats['n']} |",
        f"| distinct scores | {stats.get('distinct_scores', 0)} |",
        "",
        "## Seniority gate",
        "",
        f"{stats.get('filtered_total', 0)} matches filtered out, by requirement source:",
        "",
        "| source | filtered |",
        "|---|---:|",
    ]
    for source, count in sorted(
        stats.get("filtered_by_source", {}).items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| {source} | {count} |")
    if not stats.get("filtered_by_source"):
        lines.append("| (none) | 0 |")

    lines += [
        "",
        "## Requirement source",
        "",
        "Which layer of the fallback chain produced each match's years figure.",
        "",
        "| source | matches | share |",
        "|---|---:|---:|",
    ]
    source_total = sum(stats.get("by_source", {}).values()) or 1
    for source, count in sorted(
        stats.get("by_source", {}).items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| {source} | {count} | {100.0 * count / source_total:.1f}% |")
    if not stats.get("by_source"):
        lines.append("| (none recorded) | 0 | — |")
    lines += [
        "",
        f"{stats.get('description_rescued', 0)} match(es) had a null structured "
        "field and got their figure from the description.",
    ]

    lines += [
        "",
        "## Histogram",
        "",
        "Fixed buckets over [0, 1] so runs stay comparable.",
        "",
        "| bucket | count | |",
        "|---|---:|---|",
    ]

    peak = max(stats["buckets"]) or 1
    for i, count in enumerate(stats["buckets"]):
        low = i / N_BUCKETS
        high = (i + 1) / N_BUCKETS
        bar = "#" * round(40 * count / peak)
        lines.append(f"| {low:.1f}-{high:.1f} | {count} | `{bar}` |")

    lines += [
        "",
        f"## Top {TOP_N} by score",
        "",
        "| # | score | title | company | needs | source | job id |",
        "|---:|---:|---|---|---:|---|---:|",
    ]
    for i, entry in enumerate(stats["top"], 1):
        title = (entry["title"] or "").replace("|", "\\|")
        company = entry["company"].replace("|", "\\|")
        years = entry.get("required_years_min")
        years_str = "—" if years is None else f"{years:g}y"
        lines.append(
            f"| {i} | {entry['score']:.4f} | {title} | {company} | {years_str} "
            f"| {entry.get('seniority_source') or '—'} | {entry['job_id']} |"
        )

    lines.append("")
    return "\n".join(lines)


def compare(baseline: dict, after: dict) -> str:
    base_top = [e["job_id"] for e in baseline["top"]]
    after_top = [e["job_id"] for e in after["top"]]
    retained = set(base_top) & set(after_top)

    rows = [
        ("median", baseline["median"], after["median"]),
        ("std dev", baseline["stdev"], after["stdev"]),
        ("spread (max-min)", baseline["spread"], after["spread"]),
        ("min", baseline["min"], after["min"]),
        ("max", baseline["max"], after["max"]),
        ("mean", baseline["mean"], after["mean"]),
    ]

    out = [
        "",
        "=" * 66,
        "BASELINE vs AFTER",
        "=" * 66,
        f"{'metric':<18} {'baseline':>10} {'after':>10} {'delta':>10} {'%':>9}",
        "-" * 66,
    ]
    for label, before, now in rows:
        delta = now - before
        pct = (delta / before * 100) if before else float("nan")
        out.append(f"{label:<18} {before:>10.4f} {now:>10.4f} {delta:>+10.4f} {pct:>+8.1f}%")

    base_ties = baseline.get("ties")
    out += [
        "-" * 66,
        f"scored matches     {baseline['n']:>10} {after['n']:>10} {after['n'] - baseline['n']:>+10}",
        f"exact ties         "
        f"{('n/a' if base_ties is None else base_ties):>10} {after.get('ties', 0):>10}"
        + ("" if base_ties is None else f" {after.get('ties', 0) - base_ties:>+10}"),
        f"distinct scores    "
        f"{(baseline.get('distinct_scores') or 'n/a'):>10} "
        f"{after.get('distinct_scores', 0):>10}",
        "",
        f"seniority gate: {after.get('filtered_total', 0)} filtered "
        f"(baseline {baseline.get('filtered_total', 0)})",
    ]
    for source, count in sorted(
        after.get("filtered_by_source", {}).items(), key=lambda kv: -kv[1]
    ):
        was = (baseline.get("filtered_by_source") or {}).get(source, 0)
        out.append(f"    source={source:<8} {count:>6}   (was {was})")

    after_sources = after.get("by_source") or {}
    if after_sources:
        out += ["", "requirement source:"]
        base_sources = baseline.get("by_source") or {}
        for source, count in sorted(after_sources.items(), key=lambda kv: -kv[1]):
            was = base_sources.get(source)
            was_str = "n/a" if was is None else str(was)
            out.append(f"    {source:<12} {count:>6}   (was {was_str})")
        out.append(
            f"    api null but description found a figure: "
            f"{after.get('description_rescued', 0)}"
        )

    out += [
        "",
        f"previous top-{TOP_N} still in top-{TOP_N}: {len(retained)}/{len(base_top)}",
    ]

    dropped = [e for e in baseline["top"] if e["job_id"] not in set(after_top)]
    entered = [e for e in after["top"] if e["job_id"] not in set(base_top)]
    if dropped:
        out.append(f"  dropped out ({len(dropped)}):")
        for e in dropped:
            out.append(f"    - [{e['job_id']}] {e['score']:.4f}  {(e['title'] or '')[:58]}")
    if entered:
        out.append(f"  newly entered ({len(entered)}):")
        for e in entered:
            out.append(f"    + [{e['job_id']}] {e['score']:.4f}  {(e['title'] or '')[:58]}")
    out.append("=" * 66)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "jobscout.db"))
    parser.add_argument("--profile", default=None, help="profile name (default: first)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="markdown output path")
    parser.add_argument(
        "--visible-only", action="store_true",
        help="exclude jobs the seniority gate hid, i.e. score what you actually see",
    )
    parser.add_argument(
        "--compare", default=None, help="baseline JSON sidecar to compare against"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        profile = resolve_profile(conn, args.profile)
        rows = collect(conn, profile["id"])
    finally:
        conn.close()

    if not rows:
        print(f"ERROR: no matches stored for profile '{profile['name']}'. Run matching first.")
        return 1

    if args.visible_only:
        hidden = [r for r in rows if is_stretch(r)]
        rows = [r for r in rows if not is_stretch(r)]
        print(f"--visible-only: dropped {len(hidden)} stretch match(es), "
              f"{len(rows)} remain")
        if not rows:
            print("ERROR: every match was hidden by the seniority gate.")
            return 1

    stats = build_stats(rows)

    out_md = Path(args.out).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(stats, profile["name"]), encoding="utf-8")

    sidecar = out_md.with_suffix(".json")
    sidecar.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Profile '{profile['name']}' - {stats['n']} scored matches")
    print(
        f"  min={stats['min']:.4f}  p25={stats['p25']:.4f}  median={stats['median']:.4f}  "
        f"p75={stats['p75']:.4f}  max={stats['max']:.4f}"
    )
    print(f"  mean={stats['mean']:.4f}  stdev={stats['stdev']:.4f}  spread={stats['spread']:.4f}")
    print(f"  wrote {out_md.relative_to(REPO_ROOT)} and {sidecar.name}")

    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.exists():
            print(f"ERROR: baseline sidecar not found: {baseline_path}")
            return 1
        print(compare(json.loads(baseline_path.read_text(encoding="utf-8")), stats))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
