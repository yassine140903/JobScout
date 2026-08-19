#!/usr/bin/env python
"""Re-seed tests/eval/skills_gold.yaml from the current stored corpus.

    uv run python scripts/make_gold_set.py
    uv run python scripts/make_gold_set.py --dry-run

Hand-written labels are the expensive part of this file, so they are never
regenerated. An entry with a non-empty `expected_skills` is LABELLED: it is
carried through verbatim and kept at the top, in its existing order. Only the
unlabelled entries are replaced.

The replacement sample is stratified against the gap we actually have, not the
corpus at large - the point of a gold set is to measure where the extractor
fails, and a uniform sample spends most of its labelling budget on cases that
already work:

    12  extract ZERO skills despite a real description   - the recall gap
     4  extract well (> 8 skills)                        - regression detection
     4  clearly non-technical roles                      - expected_skills is []
                                                           and inventing any is
                                                           a failure only an
                                                           empty case catches

Deterministic: same database in, same jobs out. The excerpt is taken from the
requirements section where one is identifiable, because that is where the
skills are - the opening paragraphs are company boilerplate.

Not part of the package.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from jobscout.profiles import RuleBasedExtractor  # noqa: E402

OUT_PATH = REPO_ROOT / "tests" / "eval" / "skills_gold.yaml"
EXCERPT_CHARS = 500

# The sample, by what each slice is for.
QUOTA_ZERO_SKILLS = 12
# Of those, how many may be in a language the labeller cannot read. The recall
# gap is worst in the EURES long tail, but a gold entry nobody can label is
# worth nothing, so the tail keeps a couple of canary slots and the rest of the
# budget goes to languages that can actually be judged.
QUOTA_ZERO_CANARY = 2
READABLE_LANGUAGES = ("fr", "en", "de")
QUOTA_RICH = 4
QUOTA_NON_TECHNICAL = 4
MIN_DESCRIPTION_CHARS = 1500   # "a real description", not a stub
RICH_SKILL_THRESHOLD = 8

# Where the requirements start, per language. Checked in order; the first that
# appears past the opening boilerplate wins.
REQUIREMENT_MARKERS = (
    "profil recherché", "votre profil", "le profil", "profil :",
    "compétences requises", "compétences techniques", "compétences attendues",
    "ce que nous recherchons", "vous êtes", "vous disposez", "vous justifiez",
    "qualifications", "requirements", "your profile", "who you are",
    "what you'll bring", "what we're looking for", "skills", "experience",
    "dein profil", "ihr profil", "anforderungen", "das bringst du mit",
    "wat wij zoeken", "jouw profiel",
)

# Titles that are clearly not engineering roles. Deliberately narrow: a false
# positive here creates a gold entry labelled [] that should not be.
NON_TECHNICAL_TITLE_RE = re.compile(
    r"\b("
    r"sales|account\s+executive|account\s+manager|business\s+developer|"
    r"commercial|commercial[e]?\b|business\s+development|"
    r"customer\s+success|partnerships|"
    r"recruit|talent\s+acquisition|human\s+resources|\brh\b|"
    r"marketing|communication|content\s+manager|"
    r"finance|financial|comptab|contrôleur\s+de\s+gestion|controller|"
    r"office\s+manager|juridique|legal|"
    r"operations\s+manager|ops\s+manager|supply\s+chain|logistique|"
    r"product\s+manager|product\s+owner|chef\s+de\s+produit"
    r")\b",
    re.IGNORECASE,
)
# A technical role, in any of the corpus languages. Used twice, in opposite
# directions:
#
#   * the zero-skill slice REQUIRES it. Without that requirement the slice
#     fills with forklift drivers, dietitians and ship's captains, which
#     extract nothing because they contain nothing — correct behaviour, not a
#     recall gap, and a wasted labelling slot.
#   * the non-technical slice REFUSES it, so "Director IT Operations" does not
#     get labelled as a role that should yield no skills.
TECHNICAL_TITLE_RE = re.compile(
    r"\b("
    r"engineer|engineering|developer|software|devops|sre|architect|scientist|"   # EN
    r"data|analytics|analyst|cloud|cyber|security|backend|frontend|fullstack|"
    r"full[\s-]?stack|machine\s+learning|deep\s+learning|"
    r"programmer|database|network|infrastructure|platform|qa|test|"
    r"développeur|développeuse|ingénieur|ingénieure|informatique|réseau|"       # FR
    r"données|technique|logiciel|système|systèmes|"
    r"entwickler|entwicklerin|informatik|softwareentwicklung|datenbank|"        # DE
    r"utvecklare|systemutvecklare|"                                             # SV/NO
    r"ontwikkelaar|programmeur|systeem|"                                        # NL
    r"sviluppatore|informatica|programmatore|"                                  # IT
    r"vývojár|vývojári|softvér|softvéru|analytik|analytici|programátor|"        # SK
    r"programista|informatyk|analityk"                                          # PL
    r")\b",
    re.IGNORECASE,
)

# Acronyms, case-SENSITIVE. Lowercased they are ordinary words in the corpus
# languages - Italian "ai" (to the), "it" inside other words - and matching
# them case-insensitively pulled a vending-machine attendant into the sample.
TECHNICAL_ACRONYM_RE = re.compile(r"\b(AI|ML|IT|IA|QA|BI|ETL|SRE|DevOps)\b")


def company_name(raw: str | None) -> str:
    if not raw:
        return "(unknown)"
    raw = raw.strip()
    if raw.startswith("{"):
        import ast
        import json

        for parse in (json.loads, ast.literal_eval):
            try:
                obj = parse(raw)
            except (ValueError, SyntaxError):
                continue
            if isinstance(obj, dict) and obj.get("name"):
                return str(obj["name"]).strip()
    return raw


def is_labelled(entry: dict) -> bool:
    """A hand-written label. Only expected_skills marks one - a years value of
    null is a legitimate label, so it cannot distinguish labelled from blank."""
    return bool(entry.get("expected_skills"))


def load_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("entries", data) if isinstance(data, dict) else data
    return entries or []


def requirements_excerpt(description: str) -> tuple[str, bool]:
    """First EXCERPT_CHARS of the requirements section. Returns (text, found)."""
    body = " ".join((description or "").split())
    lowered = body.lower()
    best = None
    for marker in REQUIREMENT_MARKERS:
        idx = lowered.find(marker)
        # Past the first fifth of the text: a marker in the opening line is
        # usually the title of the whole posting, not the requirements header.
        if idx > 0 and (best is None or idx < best):
            best = idx
    if best is not None:
        return body[best:best + EXCERPT_CHARS], True
    return body[:EXCERPT_CHARS], False


def build_entry(row: sqlite3.Row, extractor: RuleBasedExtractor, category: str) -> dict:
    text = f"{row['title'] or ''}\n{row['description'] or ''}"
    excerpt, found = requirements_excerpt(row["description"] or "")
    return {
        "job_id": row["id"],
        "title": row["title"],
        "company": company_name(row["company"]),
        "source": row["source"],
        "language": row["language"],
        "seniority_hint": extractor.extract_from_text(text)["seniority"],
        # Why this entry is in the set, so a labeller knows what it is for.
        "category": category,
        "excerpt_is_requirements": found,
        "description_excerpt": excerpt,
        "expected_skills": [],
        "expected_required_years_min": None,
        "relevance": None,
        "expected_education_level": None,
    }


def candidates(conn: sqlite3.Connection) -> list[tuple[sqlite3.Row, int]]:
    """Every job with a real description, paired with its extracted skill count."""
    extractor = RuleBasedExtractor()
    rows = conn.execute(
        "SELECT id, title, company, language, source, description FROM jobs"
        " WHERE description IS NOT NULL AND LENGTH(description) >= ?"
        " ORDER BY id",
        (MIN_DESCRIPTION_CHARS,),
    ).fetchall()
    out = []
    for row in rows:
        text = f"{row['title'] or ''}\n{row['description']}"
        out.append((row, len(extractor.extract_from_text(text)["skills"])))
    return out


def take_spread(pool: list[sqlite3.Row], quota: int, key) -> list[sqlite3.Row]:
    """Round-robin across a key so one source or language cannot fill the slice."""
    buckets: dict = {}
    for row in pool:
        buckets.setdefault(key(row), []).append(row)
    picked: list[sqlite3.Row] = []
    depth = 0
    while len(picked) < quota:
        progressed = False
        for bucket_key in sorted(buckets):
            if len(picked) >= quota:
                break
            bucket = buckets[bucket_key]
            if depth < len(bucket):
                picked.append(bucket[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return picked


def select(conn: sqlite3.Connection, exclude: set[int]) -> list[tuple[sqlite3.Row, str]]:
    pool = [(r, n) for r, n in candidates(conn) if r["id"] not in exclude]

    def technical(row) -> bool:
        title = row["title"] or ""
        return bool(TECHNICAL_TITLE_RE.search(title)
                    or TECHNICAL_ACRONYM_RE.search(title))

    def non_technical(row) -> bool:
        title = row["title"] or ""
        return bool(NON_TECHNICAL_TITLE_RE.search(title)) and not technical(row)

    # A technical posting whose description yields nothing is the recall gap.
    # A non-technical one yielding nothing is the extractor being right.
    zero = [r for r, n in pool if n == 0 and technical(r)]
    rich = [r for r, n in pool if n > RICH_SKILL_THRESHOLD and not non_technical(r)]
    plain = [r for r, n in pool if non_technical(r)]

    picked: list[tuple[sqlite3.Row, str]] = []
    seen: set[int] = set()

    def add(rows, category):
        for row in rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                picked.append((row, category))

    # Zero-skill: mostly readable languages, spread over (source, language) so
    # the gap is not measured on French WTTJ alone, plus a fixed couple of
    # canaries from the long tail so a regression there stays visible.
    readable = [r for r in zero if r["language"] in READABLE_LANGUAGES]
    tail = [r for r in zero if r["language"] not in READABLE_LANGUAGES]
    add(take_spread(readable, QUOTA_ZERO_SKILLS - QUOTA_ZERO_CANARY,
                    lambda r: (r["source"], r["language"])), "zero_skills")
    add(take_spread(tail, QUOTA_ZERO_CANARY, lambda r: r["language"]),
        "zero_skills_canary")
    add(take_spread(rich, QUOTA_RICH, lambda r: r["language"]), "rich_skills")
    add(take_spread(plain, QUOTA_NON_TECHNICAL, lambda r: r["language"]),
        "non_technical")
    return picked


HEADER = """\
# Gold set for skill-extraction evaluation.
#
# Re-seeded against the post-M7d corpus: every WTTJ description here is the
# real posting text from the detail endpoint, not the ~1KB requirements blurb
# the Algolia index carries. The earlier scaffold sampled a distribution that
# no longer exists.
#
# LABELLED entries (non-empty expected_skills) are hand-written and are never
# regenerated - scripts/make_gold_set.py preserves them verbatim, at the top,
# and replaces only the unlabelled ones.
#
#   expected_skills             list of skill strings the extractor should find.
#                               [] on a non-technical role is a real label: an
#                               extractor that invents skills there is failing.
#   expected_required_years_min minimum years the posting demands, or null when
#                               it states none. null is an assertion, not a
#                               blank - the parser must return None.
#   relevance                   yes | no | borderline - is this a job worth
#                               surfacing to this profile at all?
#   expected_education_level    e.g. "bac+5", or null.
#
# `category` says why the entry is in the set: zero_skills is the recall gap,
# rich_skills is regression detection, non_technical must extract nothing.
# `seniority_hint`, `category`, `excerpt_is_requirements` and
# `description_excerpt` are context for labelling and are not scored.
#
# Score with:  uv run python tests/eval/run_eval.py
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(REPO_ROOT / "jobscout.db"))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="report the composition, write nothing")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    out_path = Path(args.out).resolve()
    existing = load_entries(out_path)
    labelled = [e for e in existing if is_labelled(e)]
    dropped = [e for e in existing if not is_labelled(e)]

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Labelled entries keep their ids; never sample one of them again.
        exclude = {
            e["job_id"] for e in labelled if isinstance(e.get("job_id"), int)
        }
        picked = select(conn, exclude)
    finally:
        conn.close()

    unresolved = [e for e in labelled if not isinstance(e.get("job_id"), int)]
    if unresolved:
        print("ERROR: labelled entries with an unresolved job_id:")
        for e in unresolved:
            print(f"    {e.get('job_id')!r}  {e.get('title')!r}")
        print("Resolve them against the database before re-seeding.")
        return 1

    extractor = RuleBasedExtractor()
    fresh = [build_entry(row, extractor, category) for row, category in picked]

    # Every entry carries the full schema, labelled ones included, without
    # touching any value already written by hand.
    for entry in labelled:
        entry.setdefault("category", "hand_labelled")
        entry.setdefault("relevance", None)
        entry.setdefault("expected_education_level", None)

    entries = labelled + fresh

    # The guard: a labelled entry must never be lost by a re-seed.
    kept = {e["job_id"] for e in entries if is_labelled(e)}
    if kept != {e["job_id"] for e in labelled}:
        print("ERROR: a labelled entry would be dropped. Refusing to write.")
        return 1

    print(f"Gold set: {len(entries)} entries "
          f"({len(labelled)} labelled, {len(fresh)} unlabelled)")
    print(f"  preserved : {len(labelled)} hand-labelled, verbatim")
    print(f"  replaced  : {len(dropped)} unlabelled scaffold entries")
    print(f"  by category : {dict(Counter(e['category'] for e in entries))}")
    print(f"  by source   : {dict(Counter(e.get('source') or 'hand' for e in entries))}")
    for category in ("zero_skills", "zero_skills_canary", "rich_skills",
                     "non_technical"):
        rows = [e for e in fresh if e["category"] == category]
        print(f"    {category:<14} {len(rows):>2}  "
              f"src={dict(Counter(e['source'] for e in rows))} "
              f"lang={dict(Counter(e['language'] for e in rows))}")
    print(f"  by language : {dict(Counter(e['language'] for e in entries))}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    payload = yaml.safe_dump(
        {"entries": entries},
        allow_unicode=True, sort_keys=False, width=100, default_flow_style=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HEADER + payload, encoding="utf-8")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
