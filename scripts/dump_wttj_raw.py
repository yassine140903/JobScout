#!/usr/bin/env python
"""Dump raw WTTJ Algolia hits to see what the adapter throws away.

Diagnostic only - reads src/jobscout/sources/wttj.py for the endpoint and
request shape, sends nothing new, writes nothing back into the package:

    uv run python scripts/dump_wttj_raw.py
    uv run python scripts/dump_wttj_raw.py --query "data engineer"

Writes the complete raw JSON of the first 3 hits to scripts/out/wttj_raw.json
and prints a key census plus an experience/seniority field report to stdout.

The Algolia search key is scrubbed from everything this script emits.

Not part of the package.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT = REPO_ROOT / "scripts" / "out" / "wttj_raw.json"
DEFAULT_QUERY = "data"
HITS_PER_PAGE = 20
DUMP_HITS = 3

TARGET_TITLE_TOKENS = ("architect", "cloud", "data")
TARGET_COMPANY = "credit agricole"
TARGET_QUERY = "Architect Cloud et Data"

# Key names that would carry an experience requirement if one existed.
INTERESTING_KEY_RE = re.compile(
    r"exper|senior|level|niveau|profile|years?|annee|xp|grade|qualif", re.I,
)
# Value shapes that look like a seniority enum even under an innocuous key name.
INTERESTING_VALUE_RE = re.compile(
    r"\d+\s*(?:_|-|\s)?(?:to|a)?[\s_-]*\d*\s*(?:years?|ans|months?|mois)"
    r"|senior|junior|intern|entry[_ -]?level|mid[_ -]?level|confirme|debutant",
    re.I,
)

MAX_VALUE_REPR = 160
MAX_DISTINCT = 12


def deaccent(text: str) -> str:
    """Fold accents so 'Credit' matches the accented spelling - WTTJ is inconsistent."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).lower()


def scrub(text: str, secrets: list[str]) -> str:
    """Never let the shipped search key reach stdout or disk."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def short(value: Any) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    if len(rendered) > MAX_VALUE_REPR:
        rendered = rendered[:MAX_VALUE_REPR] + f"... (+{len(rendered) - MAX_VALUE_REPR} chars)"
    return rendered


def hashable(value: Any) -> str:
    """Stable string form so dict/list values can go into a set."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def fetch_page(query: str, hits_per_page: int, page: int) -> list[dict[str, Any]]:
    """Same endpoint, headers, body and params the adapter itself sends."""
    import httpx

    from jobscout.sources.wttj import WTTJAdapter

    adapter = WTTJAdapter()
    with httpx.Client(timeout=30) as client:
        result = adapter._algolia_search(client, query, page, hits_per_page, None)
    return result.get("hits", [])


def collect_top_level(hits: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for hit in hits:
        counts.update(hit.keys())
    return counts


def collect_nested(hits: list[dict[str, Any]]) -> "OrderedDict[str, Counter]":
    """One level deep: dict children as parent.child, list-of-dict as parent[].child."""
    nested: OrderedDict[str, Counter] = OrderedDict()
    for hit in hits:
        for key, value in hit.items():
            if isinstance(value, dict):
                nested.setdefault(key, Counter()).update(f"{key}.{sub}" for sub in value)
            elif isinstance(value, list) and any(isinstance(e, dict) for e in value):
                bucket = nested.setdefault(f"{key}[]", Counter())
                for element in value:
                    if isinstance(element, dict):
                        bucket.update(f"{key}[].{sub}" for sub in element)
    return nested


def values_for_path(hits: list[dict[str, Any]], path: str) -> list[Any]:
    """Pull every value at a dotted path (supports one '[]' list hop)."""
    out: list[Any] = []
    for hit in hits:
        if "[]" in path:
            parent, _, child = path.partition("[].")
            parent = parent.replace("[]", "")
            container = hit.get(parent)
            if isinstance(container, list):
                for element in container:
                    if isinstance(element, dict) and child in element:
                        out.append(element[child])
        elif "." in path:
            parent, _, child = path.partition(".")
            container = hit.get(parent)
            if isinstance(container, dict) and child in container:
                out.append(container[child])
        elif path in hit:
            out.append(hit[path])
    return out


def is_interesting(path: str, values: list[Any]) -> bool:
    leaf = path.split(".")[-1]
    if INTERESTING_KEY_RE.search(leaf):
        return True
    for value in values:
        if isinstance(value, str) and INTERESTING_VALUE_RE.search(value):
            return True
    return False


def describe_format(values: list[Any]) -> str:
    """Classify what an experience-ish field actually holds."""
    present = [v for v in values if v not in (None, "", {}, [])]
    if not present:
        return "always null/empty"
    if all(isinstance(v, bool) for v in present):
        return "boolean"
    if all(isinstance(v, int) for v in present):
        return f"integer (min {min(present)}, max {max(present)})"
    if all(isinstance(v, (int, float)) for v in present):
        return "number"
    if all(isinstance(v, str) for v in present):
        distinct = set(present)
        if len(distinct) <= max(8, len(present) // 2):
            return f"enum string ({len(distinct)} distinct values)"
        return "free text string"
    if all(isinstance(v, dict) for v in present):
        keys = sorted({k for v in present for k in v})
        if {"min", "max"} & set(keys):
            return f"range object (keys: {', '.join(keys)})"
        return f"object (keys: {', '.join(keys)})"
    if all(isinstance(v, list) for v in present):
        return "list"
    return "mixed types"


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def report_interesting(
    hits: list[dict[str, Any]], paths: list[str], label: str,
) -> list[tuple[str, list[Any]]]:
    print_section(label)
    found: list[tuple[str, list[Any]]] = []
    for path in paths:
        values = values_for_path(hits, path)
        if not is_interesting(path, values):
            continue
        found.append((path, values))
        distinct: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = hashable(value)
            if key not in seen:
                seen.add(key)
                distinct.append(short(value))
        print(f"\n  {path}  ({len(values)}/{len(hits)} hits, {len(distinct)} distinct)")
        for rendered in distinct[:MAX_DISTINCT]:
            print(f"      {rendered}")
        if len(distinct) > MAX_DISTINCT:
            print(f"      ... and {len(distinct) - MAX_DISTINCT} more distinct values")
    if not found:
        print("  (none matched)")
    return found


def find_target(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    for hit in hits:
        title = deaccent(str(hit.get("name") or ""))
        org = hit.get("organization") or {}
        company = deaccent(str(org.get("name") or ""))
        if all(token in title for token in TARGET_TITLE_TOKENS):
            if not company or TARGET_COMPANY in company:
                return hit
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Algolia query string")
    parser.add_argument("--hits", type=int, default=HITS_PER_PAGE, help="hits per page")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="raw JSON dump path")
    args = parser.parse_args()

    # WTTJ text is French/UTF-8; the default Windows console codec chokes on it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    try:
        from jobscout.sources.wttj import ALGOLIA_API_KEY, ALGOLIA_URL, JOB_INDEX
    except ImportError as exc:
        print(f"ERROR: cannot import the WTTJ adapter ({exc}).", file=sys.stderr)
        print(
            "Run from the repo root, e.g. uv run python scripts/dump_wttj_raw.py",
            file=sys.stderr,
        )
        return 2

    secrets = [ALGOLIA_API_KEY]
    endpoint = scrub(ALGOLIA_URL.split("?")[0], secrets)

    print(f"endpoint : {endpoint}")
    print(f"index    : {JOB_INDEX}")
    print(f"query    : {args.query!r}   hitsPerPage={args.hits} page=0")
    print("api key  : <redacted, read from the adapter>")

    try:
        hits = fetch_page(args.query, args.hits, 0)
    except Exception as exc:  # network, DNS, TLS, HTTP status, bad JSON
        print(
            f"\nERROR: WTTJ Algolia request failed: {type(exc).__name__}: "
            f"{scrub(str(exc), secrets)}",
            file=sys.stderr,
        )
        print("No dump written. Check connectivity, then retry.", file=sys.stderr)
        return 1

    if not hits:
        print("\nNo hits returned for that query - nothing to inspect.")
        return 1

    print(f"\nfetched {len(hits)} hits")

    # 1. Raw dump of the first N hits, unfiltered.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(hits[:DUMP_HITS], ensure_ascii=False, indent=2, default=str)
    args.out.write_text(scrub(payload, secrets), encoding="utf-8")
    print(f"wrote first {min(DUMP_HITS, len(hits))} raw hits -> {args.out}")

    # 2. Top-level key census.
    top_counts = collect_top_level(hits)
    print_section(f"Top-level keys across {len(hits)} hits")
    for key, count in sorted(top_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        flag = "" if count == len(hits) else "   <- optional"
        print(f"  {count:>3}/{len(hits)}  {key}{flag}")

    # 3. Nested key census, one level deep.
    nested = collect_nested(hits)
    print_section("Nested keys (one level deep)")
    nested_paths: list[str] = []
    for parent in sorted(nested):
        print(f"\n  {parent}")
        for path, count in sorted(nested[parent].items(), key=lambda kv: (-kv[1], kv[0])):
            nested_paths.append(path)
            print(f"      {count:>3}  {path}")
    if not nested:
        print("  (no nested objects)")

    # 4. Experience / seniority candidates.
    top_found = report_interesting(
        hits, sorted(top_counts), "Experience / seniority candidates - top level",
    )
    nested_found = report_interesting(
        hits, nested_paths, "Experience / seniority candidates - nested",
    )

    # 5. The specific job this milestone is about.
    print_section("Target job: Architect Cloud et Data H/F (Credit Agricole, Montrouge)")
    target = find_target(hits)
    if target is None:
        print(f"  not in this page - searching the index for {TARGET_QUERY!r}")
        try:
            target_hits = fetch_page(TARGET_QUERY, args.hits, 0)
        except Exception as exc:
            print(
                f"  ERROR: targeted search failed: {type(exc).__name__}: "
                f"{scrub(str(exc), secrets)}",
                file=sys.stderr,
            )
            target_hits = []
        target = find_target(target_hits)
        if target is None and target_hits:
            print("  exact match not found. Closest titles returned:")
            for hit in target_hits[:10]:
                org = (hit.get("organization") or {}).get("name")
                print(f"      - {hit.get('name')!r} @ {org}")
    if target is not None:
        print(scrub(json.dumps(target, ensure_ascii=False, indent=2, default=str), secrets))
    else:
        print("  no record found for the target job.")

    # 6. Verdict.
    print_section("VERDICT: does the API expose a structured experience requirement?")
    structured: list[tuple[str, str]] = []
    for path, values in top_found + nested_found:
        fmt = describe_format(values)
        if fmt in ("always null/empty", "free text string"):
            continue
        leaf = path.split(".")[-1]
        if not INTERESTING_KEY_RE.search(leaf) and not any(
            isinstance(v, str) and INTERESTING_VALUE_RE.search(v) for v in values
        ):
            continue
        structured.append((path, fmt))

    if structured:
        print("  YES - structured field(s) present:\n")
        for path, fmt in structured:
            print(f"      {path}: {fmt}")
        print("\n  The adapter can read the requirement straight off the hit;")
        print("  no prose parsing needed for these fields.")
    else:
        print("  NO - no structured experience field carries a usable value.")
        print("  Experience lives only in the description prose, so the fix is a")
        print("  regex/LLM extraction problem, not an adapter mapping problem.")
    print("\n  (Fields reported as 'always null/empty' exist in the schema but are")
    print("   unset in this sample - check the raw dump before relying on them.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
