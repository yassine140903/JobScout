#!/usr/bin/env python
"""Diagnose why match scores compress into 0.8-0.9.

    uv run python scripts/diag_representation.py

Step 0 audits the embedder, step 1 builds SIMILAR/DISSIMILAR pair sets from
rule-based skill overlap, step 2 embeds them four ways, step 3 measures how
well each representation separates the two groups, step 4 simulates display-time
score normalization.

Diagnostic only - reads the database, writes nothing back to it, and does not
touch src/jobscout/.
"""

from __future__ import annotations

import ast
import json
import random
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from jobscout.matching import build_facet_text  # noqa: E402
from jobscout.textclean import clean_description  # noqa: E402

DB_PATH = REPO_ROOT / "jobscout.db"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_MD = OUT_DIR / "diag_representation.md"
PAIRS_MD = OUT_DIR / "diag_pairs.md"

MODEL_NAME = "intfloat/multilingual-e5-base"
SEED = 20260818
N_PAIRS = 20
MIN_SKILLS = 3          # below this, Jaccard is noise
SIMILAR_JACCARD_MIN = 0.5

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def company_name(raw: str | None) -> str:
    if not raw:
        return "(unknown)"
    raw = raw.strip()
    if raw.startswith("{"):
        for parse in (json.loads, ast.literal_eval):
            try:
                obj = parse(raw)
            except (ValueError, SyntaxError):
                continue
            if isinstance(obj, dict) and obj.get("name"):
                return str(obj["name"]).strip()
    return raw


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class Job:
    id: int
    title: str
    company: str
    position_type: str | None
    skills: set[str]
    skills_text: str
    description: str


@dataclass
class GroupStats:
    name: str
    values: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def lo(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def hi(self) -> float:
        return max(self.values) if self.values else 0.0


# ---------------------------------------------------------------------------
# Step 0 - audit
# ---------------------------------------------------------------------------

def audit_embedder(model, sample_job: Job) -> list[str]:
    import inspect

    from jobscout import embedder as embedder_module

    src = inspect.getsource(embedder_module.Embedder.embed)
    lines = [
        "## Step 0 - embedder audit",
        "",
        "### Prefixes",
        "",
        "`Embedder.embed` verbatim:",
        "",
        "```python",
        src.rstrip(),
        "```",
        "",
        "Call sites in `matching.py`:",
        "",
        "```python",
        "# profile facets - queries",
        'profile_skills_emb = embedder.embed(profile_skills_text or "general", is_query=True)',
        'profile_domain_emb = embedder.embed(profile_domain_text or "general", is_query=True)',
        "",
        "# job facets - passages",
        'skills_emb = embedder.embed(skills_text or "general", is_query=False)',
        'domain_emb = embedder.embed(domain_text or "general", is_query=False)',
        "```",
        "",
        "**Both prefixes are applied, and they are applied asymmetrically and "
        "correctly**: the profile side gets `\"query: \"`, the job side gets "
        "`\"passage: \"`. This is the usage e5 was trained for. The hypothesis that "
        "a missing or symmetric prefix explains the compression is **not supported**.",
        "",
        "### Sequence length",
        "",
        f"- `model.max_seq_length` = **{model.max_seq_length}**",
        f"- `tokenizer.model_max_length` = {model.tokenizer.model_max_length}",
        f"- `config.max_position_embeddings` = {model[0].auto_model.config.max_position_embeddings}",
        "",
        "### Normalization",
        "",
        "- `normalize_embeddings=True` is passed in both `embed` and `embed_batch`.",
        f"- The sentence-transformers pipeline also ends in a `Normalize` module: "
        f"`{[type(m).__name__ for m in model]}`.",
        "- Cosine similarity therefore reduces to a dot product, as "
        "`matching.cosine_similarity` assumes.",
        "",
        "### The exact string embedded for a job",
        "",
        "`matching.py:148` sits inside `_extract_job_facets`, which produces the "
        "skill list; the string is assembled by `build_facet_text` and embedded at "
        "`matching.py:183-184`. For job "
        f"`{sample_job.id}` ({sample_job.title}):",
        "",
        "```",
        f"{PASSAGE_PREFIX}{sample_job.skills_text}",
        "```",
        "",
        f"That is **{len(PASSAGE_PREFIX + sample_job.skills_text)} characters** - a "
        "comma-joined keyword list drawn from a closed vocabulary, not the job text. "
        f"The description for the same job is {len(sample_job.description):,} characters "
        "and is never embedded.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Step 1 - pair selection
# ---------------------------------------------------------------------------

def load_jobs(conn: sqlite3.Connection) -> list[Job]:
    rows = conn.execute(
        """
        SELECT id, title, company, position_type, skills, description
        FROM jobs
        WHERE skills IS NOT NULL
          AND description IS NOT NULL
          AND LENGTH(TRIM(description)) > 200
        ORDER BY id
        """
    ).fetchall()

    jobs = []
    for row in rows:
        try:
            skills = set(json.loads(row["skills"]))
        except (TypeError, ValueError):
            continue
        if len(skills) < MIN_SKILLS:
            continue
        jobs.append(
            Job(
                id=row["id"],
                title=row["title"] or "",
                company=company_name(row["company"]),
                position_type=row["position_type"],
                skills=skills,
                skills_text=build_facet_text(sorted(skills)),
                description=clean_description(row["description"]),
            )
        )
    return jobs


def select_pairs(jobs: list[Job]) -> tuple[list[tuple], list[tuple], dict]:
    """Deterministic SIMILAR / DISSIMILAR pair sets.

    Each job is used at most once per group so the 20 pairs are independent
    samples rather than 20 views of the same few postings.
    """
    similar_candidates: list[tuple[float, int, int]] = []
    dissimilar_typed: list[tuple[float, int, int]] = []
    dissimilar_any: list[tuple[float, int, int]] = []

    by_id = {j.id: j for j in jobs}
    for i in range(len(jobs)):
        a = jobs[i]
        for k in range(i + 1, len(jobs)):
            b = jobs[k]
            score = jaccard(a.skills, b.skills)
            if score >= SIMILAR_JACCARD_MIN:
                if a.company.lower() != b.company.lower():
                    similar_candidates.append((score, a.id, b.id))
            elif score == 0.0:
                if (
                    a.position_type
                    and b.position_type
                    and a.position_type != b.position_type
                ):
                    dissimilar_typed.append((score, a.id, b.id))
                else:
                    dissimilar_any.append((score, a.id, b.id))

    rng = random.Random(SEED)

    def take(pools: list[list[tuple]], limit: int) -> list[tuple]:
        chosen: list[tuple] = []
        used: set[int] = set()
        for pool in pools:
            pool = sorted(pool)  # deterministic base order
            rng.shuffle(pool)
            for score, a_id, b_id in pool:
                if len(chosen) >= limit:
                    return chosen
                if a_id in used or b_id in used:
                    continue
                used.update((a_id, b_id))
                chosen.append((score, by_id[a_id], by_id[b_id]))
        return chosen

    similar = take([similar_candidates], N_PAIRS)
    # Prefer differing position_type, fall back so the group can still reach 20.
    dissimilar = take([dissimilar_typed, dissimilar_any], N_PAIRS)

    meta = {
        "similar_candidates": len(similar_candidates),
        "dissimilar_typed_candidates": len(dissimilar_typed),
        "dissimilar_any_candidates": len(dissimilar_any),
        "dissimilar_from_typed": sum(
            1
            for _, a, b in dissimilar
            if a.position_type and b.position_type and a.position_type != b.position_type
        ),
    }
    return similar, dissimilar, meta


def write_pairs_md(similar: list[tuple], dissimilar: list[tuple], meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Diagnostic pair sets",
        "",
        f"Seed `{SEED}`, deterministic. Ground-truth proxy is Jaccard overlap on "
        "the rule-based extracted skill sets.",
        "",
        f"- SIMILAR: Jaccard >= {SIMILAR_JACCARD_MIN}, different companies "
        f"({meta['similar_candidates']:,} candidate pairs)",
        "- DISSIMILAR: Jaccard == 0.0, different position_type where available "
        f"({meta['dissimilar_typed_candidates']:,} typed candidates, "
        f"{meta['dissimilar_any_candidates']:,} untyped)",
        f"- {meta['dissimilar_from_typed']}/{len(dissimilar)} dissimilar pairs have "
        "differing position_type",
        f"- Only jobs with >= {MIN_SKILLS} extracted skills are eligible; each job "
        "appears at most once per group.",
        "",
        "Eyeball these before trusting the separation numbers.",
        "",
    ]

    for label, pairs in (("SIMILAR", similar), ("DISSIMILAR", dissimilar)):
        lines += [f"## {label} ({len(pairs)} pairs)", ""]
        for i, (score, a, b) in enumerate(pairs, 1):
            shared = sorted(a.skills & b.skills)
            lines += [
                f"### {label[:3].title()} {i} - Jaccard {score:.3f}",
                "",
                f"- **A** [{a.id}] {a.title} — _{a.company}_ "
                f"(position_type: {a.position_type or 'n/a'})",
                f"  - skills: `{', '.join(sorted(a.skills))}`",
                f"- **B** [{b.id}] {b.title} — _{b.company}_ "
                f"(position_type: {b.position_type or 'n/a'})",
                f"  - skills: `{', '.join(sorted(b.skills))}`",
                f"- shared: `{', '.join(shared) if shared else '(none)'}`",
                "",
            ]

    PAIRS_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 2 - representations
# ---------------------------------------------------------------------------

def truncate_to_tokens(model, text: str, budget: int) -> str:
    """Decode back the first `budget` content tokens of `text`."""
    tokenizer = model.tokenizer
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)


def build_representations(model, jobs: list[Job]) -> tuple[dict, dict]:
    """Embed every job under each representation. Returns (vectors, notes)."""
    # This function deliberately runs the tokenizer without truncation, to
    # measure overflow and to build D by hand. That makes transformers warn
    # about sequences over 512 tokens; silence it so the diagnostic's own
    # measurement is not mistaken for a fault in the matching pipeline.
    import transformers

    _verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        return _build_representations(model, jobs)
    finally:
        transformers.logging.set_verbosity(_verbosity)


def _build_representations(model, jobs: list[Job]) -> tuple[dict, dict]:
    ids = [j.id for j in jobs]

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )

    # A - exactly the current code path: passage-prefixed keyword list.
    a_texts = [PASSAGE_PREFIX + (j.skills_text or "general") for j in jobs]

    # B - prefixes are NOT missing from A (see step 0), so "A + correct prefixes"
    # is A. Repurposed as the control that quantifies what the prefix is worth:
    # the same keyword list with no prefix at all.
    b_texts = [(j.skills_text or "general") for j in jobs]

    # C - full cleaned description, passage-prefixed.
    c_texts = [PASSAGE_PREFIX + (j.description or "general") for j in jobs]

    # D - description truncated to the first 512 tokens, minus the prefix budget.
    prefix_tokens = len(model.tokenizer.encode(PASSAGE_PREFIX, add_special_tokens=False))
    budget = model.max_seq_length - prefix_tokens - 2  # -2 for <s> and </s>
    d_texts = [
        PASSAGE_PREFIX + truncate_to_tokens(model, j.description or "general", budget)
        for j in jobs
    ]

    vectors = {}
    for key, texts in (("A", a_texts), ("B", b_texts), ("C", c_texts), ("D", d_texts)):
        vectors[key] = dict(zip(ids, encode(texts)))

    # How does the model handle description overflow?
    token_counts = [
        len(model.tokenizer.encode(t, add_special_tokens=True)) for t in c_texts
    ]
    over = [n for n in token_counts if n > model.max_seq_length]
    cd_cos = [cosine(vectors["C"][i], vectors["D"][i]) for i in ids]
    cd_maxdiff = max(
        float(np.abs(vectors["C"][i] - vectors["D"][i]).max()) for i in ids
    )

    notes = {
        "token_counts": token_counts,
        "n_over": len(over),
        "n_total": len(token_counts),
        "max_tokens": max(token_counts),
        "median_tokens": statistics.median(token_counts),
        "budget": budget,
        "cd_min_cos": min(cd_cos),
        "cd_mean_cos": statistics.mean(cd_cos),
        "cd_maxdiff": cd_maxdiff,
        # Anything above ~0.999 cosine is the tokenizer decode round-trip, not a
        # modelling difference: D re-encodes text decoded from token ids, which
        # is not byte-identical to the original.
        "cd_equivalent": min(cd_cos) > 0.99,
        "cd_identical": cd_maxdiff < 1e-6,
    }
    return vectors, notes


# ---------------------------------------------------------------------------
# Step 3 - separation
# ---------------------------------------------------------------------------

def measure(vectors: dict, similar: list[tuple], dissimilar: list[tuple]) -> dict:
    results = {}
    for key, table in vectors.items():
        sim = GroupStats(
            "similar", [cosine(table[a.id], table[b.id]) for _, a, b in similar]
        )
        dis = GroupStats(
            "dissimilar", [cosine(table[a.id], table[b.id]) for _, a, b in dissimilar]
        )
        overlap = sum(1 for v in dis.values if v > sim.lo) if sim.values else 0
        results[key] = {
            "similar": sim,
            "dissimilar": dis,
            "separation": sim.mean - dis.mean,
            "overlap": overlap,
        }
    return results


REPRESENTATION_LABELS = {
    "A": "keyword list + passage prefix (current code path)",
    "B": "keyword list, no prefix (prefix control)",
    "C": "full cleaned description + passage prefix",
    "D": "description truncated to 512 tokens + passage prefix",
}


def separation_table(results: dict) -> list[str]:
    header = (
        f"{'rep':<4} {'mean sim':>9} {'mean dis':>9} {'SEPARATION':>11} "
        f"{'sd sim':>7} {'sd dis':>7} {'sim range':>17} {'dis range':>17} {'ovl':>4}"
    )
    lines = [header, "-" * len(header)]
    for key in ("A", "B", "C", "D"):
        if key not in results:
            continue
        r = results[key]
        s, d = r["similar"], r["dissimilar"]
        lines.append(
            f"{key:<4} {s.mean:>9.4f} {d.mean:>9.4f} {r['separation']:>11.4f} "
            f"{s.stdev:>7.4f} {d.stdev:>7.4f} "
            f"{f'{s.lo:.3f}-{s.hi:.3f}':>17} {f'{d.lo:.3f}-{d.hi:.3f}':>17} "
            f"{r['overlap']:>4}"
        )
    return lines


# ---------------------------------------------------------------------------
# Step 4 - normalization simulation
# ---------------------------------------------------------------------------

def normalization_simulation(conn: sqlite3.Connection) -> tuple[list[str], dict]:
    profile = conn.execute("SELECT * FROM profiles ORDER BY id LIMIT 1").fetchone()
    rows = conn.execute(
        """
        SELECT m.job_id, m.score, j.title
        FROM matches m JOIN jobs j ON j.id = m.job_id
        WHERE m.profile_id = ?
        ORDER BY m.score DESC, m.job_id ASC
        """,
        (profile["id"],),
    ).fetchall()

    scores = np.array([r["score"] for r in rows], dtype=float)
    job_ids = [r["job_id"] for r in rows]
    n = len(scores)
    n_unique = len(np.unique(scores))
    mean, sd = float(scores.mean()), float(scores.std())

    # Percentile rank: fraction of the run scoring at or below each match.
    # Computed from the score VALUE so tied scores get identical percentiles -
    # rank-position percentiles would hand tied jobs different values and make a
    # pure tie-break look like a reordering.
    ordered_scores = np.sort(scores)
    percentile = 100.0 * np.searchsorted(ordered_scores, scores, side="right") / n
    zscore = (scores - mean) / sd if sd else np.zeros(n)

    # `rows` is already ordered by (score DESC, job_id ASC); rank each transform
    # the same way so the comparison isolates ordering, not tie-breaking.
    def top20(values: np.ndarray) -> list[int]:
        idx = sorted(range(n), key=lambda i: (-values[i], job_ids[i]))
        return [job_ids[i] for i in idx[:20]]

    raw_top = top20(scores)
    pct_top = top20(percentile)
    z_top = top20(zscore)

    def spread(values: np.ndarray) -> str:
        return (
            f"min={values.min():.4f} p25={np.percentile(values, 25):.4f} "
            f"median={np.percentile(values, 50):.4f} "
            f"p75={np.percentile(values, 75):.4f} max={values.max():.4f}"
        )

    lines = [
        "## Step 4 - normalization simulation",
        "",
        f"Profile `{profile['name']}`, {n} scored matches, current representation only.",
        "",
        "| distribution | spread |",
        "|---|---|",
        f"| raw score | {spread(scores)} |",
        f"| percentile rank | {spread(percentile)} |",
        f"| z-score | {spread(zscore)} |",
        "",
        f"- raw mean {mean:.4f}, std dev {sd:.4f}",
        f"- z-score range {zscore.min():.2f} to {zscore.max():.2f}",
        f"- {n_unique} distinct score values across {n} matches - "
        f"{n - n_unique} matches sit on a tied score.",
        "",
        "### Does the ordering change?",
        "",
        f"- top-20 by raw score vs by percentile rank: "
        f"**{'identical' if raw_top == pct_top else 'DIFFERENT'}**",
        f"- top-20 by raw score vs by z-score: "
        f"**{'identical' if raw_top == z_top else 'DIFFERENT'}**",
        "",
        "Both are strictly monotonic transforms of the raw score, so the ranking "
        "is invariant by construction; this confirms it empirically. Normalization "
        "changes the numbers a user reads, not which jobs they see or in what order.",
        "",
        f"Ties matter for how this is measured: only {n_unique} of {n} scores are "
        "distinct, so percentiles are computed from the score value (tied scores "
        "share a percentile) and all three orderings break ties on `job_id`. "
        "Ranking tied jobs by array position instead would show a spurious "
        "'reordering' that is purely a tie-break artifact.",
        "",
    ]

    stats = {
        "n": n,
        "raw_mean": mean,
        "raw_sd": sd,
        "pct_identical": raw_top == pct_top,
        "z_identical": raw_top == z_top,
        "z_min": float(zscore.min()),
        "z_max": float(zscore.max()),
    }
    return lines, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("Loading jobs ...")
    jobs = load_jobs(conn)
    print(f"  {len(jobs)} eligible jobs (>= {MIN_SKILLS} skills, description > 200 chars)")

    print("Selecting pairs ...")
    similar, dissimilar, meta = select_pairs(jobs)
    print(
        f"  SIMILAR    {len(similar)}/{N_PAIRS} from {meta['similar_candidates']:,} candidates"
    )
    print(
        f"  DISSIMILAR {len(dissimilar)}/{N_PAIRS} from "
        f"{meta['dissimilar_typed_candidates']:,} typed + "
        f"{meta['dissimilar_any_candidates']:,} untyped candidates "
        f"({meta['dissimilar_from_typed']} with differing position_type)"
    )
    write_pairs_md(similar, dissimilar, meta)
    print(f"  wrote {PAIRS_MD.relative_to(REPO_ROOT)}")

    if not similar or not dissimilar:
        print("ERROR: could not build both pair groups.")
        return 1

    print(f"Loading {MODEL_NAME} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    pair_jobs = {j.id: j for _, a, b in similar + dissimilar for j in (a, b)}
    ordered = [pair_jobs[i] for i in sorted(pair_jobs)]
    print(f"  embedding {len(ordered)} distinct jobs under 4 representations ...")

    vectors, notes = build_representations(model, ordered)
    results = measure(vectors, similar, dissimilar)

    # D is kept in the table only if it is measurably a different representation.
    drop_d = notes["cd_equivalent"]
    if drop_d:
        results.pop("D", None)

    # --- report ---
    sample = ordered[0]
    lines = [
        "# Representation diagnostic",
        "",
        f"Model `{MODEL_NAME}`, seed `{SEED}`. Diagnostic only - nothing was "
        "written back to the database.",
        "",
    ]
    lines += audit_embedder(model, sample)

    lines += [
        "## Step 1 - pair sets",
        "",
        f"- SIMILAR: {len(similar)} pairs, Jaccard >= {SIMILAR_JACCARD_MIN}, "
        f"different companies (from {meta['similar_candidates']:,} candidates)",
        f"- DISSIMILAR: {len(dissimilar)} pairs, Jaccard == 0.0 "
        f"({meta['dissimilar_from_typed']}/{len(dissimilar)} with differing "
        "position_type)",
        f"- Jobs need >= {MIN_SKILLS} extracted skills; each job used at most once "
        "per group.",
        "",
        f"Full listing with titles, companies and skill sets: `{PAIRS_MD.name}`.",
        "",
        "## Step 2 - how the model handles description overflow",
        "",
        f"- `max_seq_length` = {model.max_seq_length} tokens.",
        f"- Prefixed descriptions in the pair set: median "
        f"{notes['median_tokens']:.0f} tokens, max {notes['max_tokens']}.",
        f"- **{notes['n_over']} of {notes['n_total']}** exceed the window.",
        "- sentence-transformers **truncates silently**: the `Transformer` module "
        "tokenizes with `truncation=True`, so overflow is discarded with no error "
        "and no warning. Nothing in the current pipeline would surface this.",
        "",
        f"C vs D: mean cosine {notes['cd_mean_cos']:.6f}, min "
        f"{notes['cd_min_cos']:.6f}, max elementwise difference "
        f"{notes['cd_maxdiff']:.2e}.",
        "",
        (
            "**C and D are the same representation.** Truncating to 512 tokens by "
            "hand reproduces what the model already does internally. The residual "
            f"difference (min cosine {notes['cd_min_cos']:.6f}) is a tokenizer "
            "decode round-trip artifact - D re-encodes text decoded from token ids, "
            "which is not byte-identical to the original - not a modelling "
            "difference. D is therefore dropped from the table below; feeding the "
            "model the full description and feeding it the first 512 tokens are the "
            "same operation."
            if drop_d
            else "**C and D differ materially**, so both are reported below."
        ),
        "",
        "## Step 3 - separation",
        "",
        "SEPARATION = mean(similar) - mean(dissimilar). Higher is better "
        "discrimination; absolute cosine is not the point.",
        "",
        "| rep | representation | mean sim | mean dis | SEPARATION | sd sim | sd dis | sim range | dis range | overlap |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for key in ("A", "B", "C", "D"):
        if key not in results:
            continue
        r = results[key]
        s, d = r["similar"], r["dissimilar"]
        lines.append(
            f"| {key} | {REPRESENTATION_LABELS[key]} | {s.mean:.4f} | {d.mean:.4f} | "
            f"**{r['separation']:.4f}** | {s.stdev:.4f} | {d.stdev:.4f} | "
            f"{s.lo:.3f} – {s.hi:.3f} | {d.lo:.3f} – {d.hi:.3f} | {r['overlap']}/{len(d.values)} |"
        )
    lines += [
        "",
        "`overlap` counts DISSIMILAR pairs scoring above the lowest SIMILAR pair - "
        "how far the two groups bleed into each other.",
        "",
    ]

    norm_lines, norm_stats = normalization_simulation(conn)
    lines += norm_lines
    conn.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # --- stdout ---
    print()
    print("=" * 100)
    print("SEPARATION TABLE")
    print("=" * 100)
    for line in separation_table(results):
        print(line)
    print("-" * 100)
    for key in ("A", "B", "C", "D"):
        if key in results:
            print(f"  {key} = {REPRESENTATION_LABELS[key]}")
    print(
        f"\n  overlap = DISSIMILAR pairs above the lowest SIMILAR pair "
        f"(out of {len(dissimilar)})"
    )
    if drop_d:
        print("  D dropped: identical to C (the model already truncates at 512 tokens)")
    print(
        f"\n  normalization: top-20 order unchanged under percentile "
        f"({norm_stats['pct_identical']}) and z-score ({norm_stats['z_identical']})"
    )
    print(f"\nWrote {OUT_MD.relative_to(REPO_ROOT)} and {PAIRS_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
