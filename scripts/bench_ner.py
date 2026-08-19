#!/usr/bin/env python
"""Throwaway benchmark: local NER skill extraction vs. the rule-based extractor.

Answers one question: is `jjzha/escoxlmr_knowledge_extraction` fast enough on CPU
to replace RuleBasedExtractor for skill extraction?

Outputs:
  - stdout: latency table over batch_size x truncation, peak RSS, model load time
  - scripts/out/ner_spans.md: per-job NER spans vs. rule-based skills, for manual review

Run with:  uv run python scripts/bench_ner.py

NOT part of the package. Nothing under src/jobscout/ is modified; RuleBasedExtractor
is imported read-only as the comparison baseline.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Importable even if the package is not installed into the active venv.
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_ID = "jjzha/escoxlmr_knowledge_extraction"
DB_PATH = REPO_ROOT / "jobscout.db"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_MD = OUT_DIR / "ner_spans.md"

N_JOBS = 20
MAX_TOKENS = 128          # matches the model's training sequence length
MIN_SENTENCE_CHARS = 15   # drop fragments shorter than this
BATCH_SIZES = (8, 16, 32)
TRUNCATIONS: tuple[tuple[str, int | None], ...] = (("full", None), ("2000ch", 2000))
EXTRAPOLATE_TO = 500

SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
WS_RE = re.compile(r"\s+")

# Trimmed off the edges of an emitted span. Sentencepiece groups trailing
# punctuation into the same whitespace-word ("Python," is one word), so spans
# come out with it attached. Deliberately excludes . + # - / so ".NET", "C++",
# "C#" and "CI/CD" survive intact.
SPAN_STRIP_CHARS = " \t\r\n,;:!?()[]{}<>\"'`«»*|"


# ---------------------------------------------------------------------------
# Peak RSS
# ---------------------------------------------------------------------------

def peak_rss_mb() -> tuple[float | None, str]:
    """Peak resident set size in MB, plus the source used to measure it.

    `resource` is POSIX-only, so on Windows we fall back to the Win32
    PeakWorkingSetSize, which is the same quantity under a different name.
    """
    try:
        import resource  # POSIX only
    except ImportError:
        pass
    else:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS.
        divisor = 1024 if sys.platform.startswith("linux") else 1024 * 1024
        return ru.ru_maxrss / divisor, "resource.getrusage(ru_maxrss)"

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # argtypes/restype must be declared: the default int restype truncates the
        # GetCurrentProcess pseudo-handle on 64-bit and the call fails.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None, "unavailable"
        return counters.PeakWorkingSetSize / (1024 * 1024), "Win32 PeakWorkingSetSize"
    except Exception:
        return None, "unavailable"


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------

def company_name(raw: str | None) -> str:
    """Some adapters store the whole org object in `company`; pull the name out."""
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


def load_jobs(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Longest `limit` descriptions, so timings are worst-case not best-case."""
    return conn.execute(
        """
        SELECT id, title, company, description, LENGTH(description) AS n_chars
        FROM jobs
        WHERE description IS NOT NULL AND LENGTH(TRIM(description)) > 0
        ORDER BY n_chars DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def load_cv(conn: sqlite3.Connection) -> tuple[str | None, str]:
    """Return (raw_text, status_message). raw_text is None when unavailable."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(profiles)")}
    if "raw_text" not in cols:
        return None, "profiles.raw_text not found - CV benchmark skipped"

    row = conn.execute(
        """
        SELECT name, raw_text FROM profiles
        WHERE raw_text IS NOT NULL AND LENGTH(TRIM(raw_text)) > 0
        ORDER BY id LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None, (
            "profiles.raw_text column exists but every row is NULL/empty "
            "- CV benchmark skipped"
        )
    return row["raw_text"], (
        f"profiles.raw_text found - using profile '{row['name']}' "
        f"({len(row['raw_text']):,} chars)"
    )


def _pct(sorted_vals: list, q: float):
    """Nearest-rank percentile - no interpolation, so values stay real observations."""
    if not sorted_vals:
        return 0.0
    rank = int(-(-q * len(sorted_vals) // 1))  # ceil
    return sorted_vals[max(0, min(len(sorted_vals) - 1, rank - 1))]


def describe_lengths(label: str, lengths: list[int]) -> None:
    srt = sorted(lengths)
    print(f"  {label}: n={len(srt)} total={sum(srt):,} chars")
    print(
        f"    min={srt[0]}  p25={_pct(srt, 0.25)}  median={_pct(srt, 0.50)}  "
        f"p75={_pct(srt, 0.75)}  max={srt[-1]}  mean={statistics.mean(srt):.0f}"
    )


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Stdlib-only split. Fragments under MIN_SENTENCE_CHARS are dropped."""
    out = []
    for chunk in SENTENCE_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if len(chunk) >= MIN_SENTENCE_CHARS:
            out.append(chunk)
    return out


def normalize(text: str) -> str:
    """Comparison key: lowercased, whitespace-collapsed, edge punctuation stripped."""
    return WS_RE.sub(" ", text.lower()).strip(" .,;:()[]/-•")


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

class NerExtractor:
    """Token classification with hand-rolled BIO span aggregation.

    We deliberately do NOT use `pipeline(aggregation_strategy="simple")`. This
    checkpoint's id2label is bare {B, I, O} with no `-TYPE` suffix, and the
    pipeline's `get_tag()` reads any label lacking a `B-`/`I-` prefix as a
    *continuation* of an entity type named after the label itself. "B" and "I"
    therefore become two different entity types and every B-I-I span is torn
    into fragments. Aggregating the argmax ourselves against the fast
    tokenizer's char offsets avoids that entirely.
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "a fast tokenizer is required for char offset mapping, got a slow one"
            )
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
        self.model.to("cpu")
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def extract(self, text: str, batch_size: int) -> tuple[list[str], int]:
        """Full pipeline: split -> tokenize -> forward -> BIO aggregation.

        Returns (spans deduplicated with order preserved, n_sentences).
        """
        sentences = split_sentences(text)
        seen: dict[str, str] = {}

        for start in range(0, len(sentences), batch_size):
            for span in self._run_batch(sentences[start : start + batch_size]):
                key = normalize(span)
                if key and key not in seen:
                    seen[key] = span

        return list(seen.values()), len(sentences)

    def _run_batch(self, batch: list[str]) -> list[str]:
        if not batch:
            return []

        enc = self.tokenizer(
            batch,
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        special = enc.pop("special_tokens_mask")

        with self.torch.inference_mode():
            logits = self.model(**enc).logits
        preds = logits.argmax(dim=-1)

        attention = enc["attention_mask"]
        spans: list[str] = []
        for i, sentence in enumerate(batch):
            words = self._merge_subwords(
                preds[i].tolist(),
                offsets[i].tolist(),
                attention[i].tolist(),
                special[i].tolist(),
                enc.word_ids(i),
            )
            spans.extend(self._aggregate_bio(sentence, words))
        return spans

    def _merge_subwords(
        self,
        preds: list[int],
        offsets: list,
        attention: list[int],
        special: list[int],
        word_ids: list,
    ) -> list[tuple[int, int, str]]:
        """Collapse subword tokens into whole words, keeping each word's first label.

        Without this, "distributed" -> ["distribu", "ted"] can be tagged B,B and a
        naive per-token walk splits it into two spans mid-word. A word's label is
        the label of its first subword, which is the standard `first` aggregation.
        """
        words: list[list] = []
        prev_word_id = None

        for pred, offset, attn, is_special, word_id in zip(
            preds, offsets, attention, special, word_ids
        ):
            if not attn or is_special or word_id is None:
                continue
            start, end = int(offset[0]), int(offset[1])
            if end <= start:  # zero-width token, carries no text
                continue

            if word_id != prev_word_id:
                words.append([start, end, self.id2label.get(pred, "O")])
                prev_word_id = word_id
            else:
                words[-1][1] = end  # same word: extend, keep the first label

        return [(w[0], w[1], w[2]) for w in words]

    def _aggregate_bio(
        self, sentence: str, words: list[tuple[int, int, str]]
    ) -> list[str]:
        spans: list[str] = []
        cur: list[int] | None = None

        for start, end, label in words:
            if label == "B":
                if cur is not None:
                    spans.append(sentence[cur[0] : cur[1]])
                cur = [start, end]
            elif label == "I":
                if cur is None:
                    cur = [start, end]  # orphan I with no preceding B: start anyway
                else:
                    cur[1] = end
            else:  # "O"
                if cur is not None:
                    spans.append(sentence[cur[0] : cur[1]])
                cur = None

        if cur is not None:
            spans.append(sentence[cur[0] : cur[1]])

        return [t for t in (s.strip(SPAN_STRIP_CHARS) for s in spans) if t]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int | None) -> str:
    return text if limit is None else text[:limit]


def run_combo(
    ner: NerExtractor, texts: list[str], batch_size: int, limit: int | None
) -> dict:
    per_job: list[float] = []
    total_sentences = 0
    spans_by_index: list[list[str]] = []

    wall_start = time.perf_counter()
    for text in texts:
        t0 = time.perf_counter()
        spans, n_sentences = ner.extract(truncate(text, limit), batch_size)
        per_job.append(time.perf_counter() - t0)
        total_sentences += n_sentences
        spans_by_index.append(spans)
    wall = time.perf_counter() - wall_start

    mean = statistics.mean(per_job)
    return {
        "batch_size": batch_size,
        "trunc": "full" if limit is None else f"{limit}ch",
        "wall": wall,
        "mean": mean,
        "p95": _pct(sorted(per_job), 0.95),
        "sentences": total_sentences,
        "sent_per_sec": total_sentences / wall if wall else 0.0,
        "extrapolated_min": mean * EXTRAPOLATE_TO / 60.0,
        "spans": spans_by_index,
    }


# ---------------------------------------------------------------------------
# Qualitative report
# ---------------------------------------------------------------------------

def dedup_map(items: list[str]) -> dict[str, str]:
    """Normalized key -> original text, order preserved."""
    out: dict[str, str] = {}
    for item in items:
        key = normalize(item)
        if key and key not in out:
            out[key] = item
    return out


def md_list(items: list[str]) -> str:
    return ", ".join(f"`{i}`" for i in items) if items else "_(none)_"


def write_report(
    jobs: list[sqlite3.Row],
    job_spans: list[list[str]],
    job_rules: list[list[str]],
    cv: tuple[str, list[str], list[str]] | None,
    cv_status: str,
    combo_label: str,
) -> tuple[int, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = {"ner": 0, "rules": 0}
    lines: list[str] = [
        "# NER spans vs. rule-based skills",
        "",
        f"- Model: `{MODEL_ID}` (CPU, max {MAX_TOKENS} tokens/sentence)",
        f"- Spans taken from the **{combo_label}** run",
        "- Baseline: `RuleBasedExtractor.extract_from_text(text)['skills']`",
        f"- CV: {cv_status}",
        "",
        "Set differences are **exact normalized-string** comparisons (lowercased, "
        "whitespace-collapsed, edge punctuation stripped). A multi-word NER span such "
        "as `Python programming` will not match the rule skill `python`, so it lands "
        "in NER-only. The final subsection filters those out so you can read straight "
        "for skills the keyword vocabulary genuinely misses.",
        "",
        "---",
        "",
    ]

    def section(header: str, meta: str, ner_spans: list[str], rules: list[str]) -> None:
        ner_map = dedup_map(ner_spans)
        rule_map = dedup_map(rules)
        totals["ner"] += len(ner_map)
        totals["rules"] += len(rule_map)

        ner_only = [v for k, v in ner_map.items() if k not in rule_map]
        rules_only = [v for k, v in rule_map.items() if k not in ner_map]
        overlapping = {
            v
            for k, v in ner_map.items()
            if k not in rule_map and any(rk in k for rk in rule_map)
        }
        novel = [v for v in ner_only if v not in overlapping]

        lines.extend(
            [
                f"## {header}",
                "",
                f"_{meta}_",
                "",
                f"**NER spans ({len(ner_map)}):** {md_list(list(ner_map.values()))}",
                "",
                f"**Rule-based skills ({len(rule_map)}):** {md_list(list(rule_map.values()))}",
                "",
                f"**NER-only ({len(ner_only)}):** {md_list(ner_only)}",
                "",
                f"**Rules-only ({len(rules_only)}):** {md_list(rules_only)}",
                "",
                f"**NER-only with no lexical overlap with the rule vocabulary "
                f"({len(novel)}):** {md_list(novel)}",
                "",
                "---",
                "",
            ]
        )

    for i, (job, spans, rules) in enumerate(zip(jobs, job_spans, job_rules), 1):
        section(
            f"{i}. {job['title']}",
            f"Company: {company_name(job['company'])} - job id {job['id']} - "
            f"{job['n_chars']:,} chars - "
            f"{len(split_sentences(job['description']))} sentences",
            spans,
            rules,
        )

    if cv is not None:
        cv_text, cv_spans, cv_rules = cv
        section(
            "CV (stored profile)",
            f"{len(cv_text):,} chars - {len(split_sentences(cv_text))} sentences",
            cv_spans,
            cv_rules,
        )
    else:
        lines.extend(["## CV", "", f"**Skipped.** {cv_status}", ""])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    return totals["ner"], totals["rules"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("NER skill-extraction benchmark")
    print("=" * 78)

    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}")
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        jobs = load_jobs(conn, N_JOBS)
        cv_text, cv_status = load_cv(conn)
    finally:
        conn.close()

    if not jobs:
        print("ERROR: no job descriptions in the database.")
        return 1
    if len(jobs) < N_JOBS:
        print(f"NOTE: only {len(jobs)} jobs have descriptions (wanted {N_JOBS}).")

    job_texts = [job["description"] for job in jobs]
    print(f"\nInput data - longest {len(jobs)} descriptions (worst case):")
    describe_lengths("full", [job["n_chars"] for job in jobs])
    describe_lengths("truncated to 2000ch", [min(job["n_chars"], 2000) for job in jobs])
    print("    sentences/job (full):", [len(split_sentences(t)) for t in job_texts])

    print(f"\nCV: {cv_status}")

    print(f"\nLoading `{MODEL_ID}` on CPU.")
    print("  NOTE: the first run downloads ~2.2 GB (XLM-R-large) into the HF cache;")
    print("  later runs load from cache. Allow several minutes for a cold start.")
    load_start = time.perf_counter()
    try:
        ner = NerExtractor()
    except Exception as exc:
        print("\n" + "!" * 78)
        print("ERROR: could not load the model - benchmark aborted.")
        print(f"  {type(exc).__name__}: {exc}")
        print("\n  Common causes:")
        print("   - no network, or the ~2.2 GB download was interrupted (rerun to resume)")
        print("   - not enough free disk space in the HF cache directory")
        print("   - HF Hub rate limiting (set HF_TOKEN and retry)")
        print("!" * 78)
        return 1
    load_time = time.perf_counter() - load_start

    import torch

    print(f"  loaded in {load_time:.1f}s")
    print(f"  labels: {ner.id2label}")
    print(f"  torch {torch.__version__}, threads={torch.get_num_threads()}, device=cpu")

    # Warmup - excluded from every timing below.
    print("\nWarmup batch (excluded from timings)...")
    warm_start = time.perf_counter()
    ner.extract(job_texts[0][:2000], batch_size=8)
    print(f"  warmup took {time.perf_counter() - warm_start:.2f}s")

    results = []
    print()
    for limit_label, limit in TRUNCATIONS:
        for batch_size in BATCH_SIZES:
            print(
                f"Running batch_size={batch_size:<3} truncation={limit_label} ...",
                end="",
                flush=True,
            )
            res = run_combo(ner, job_texts, batch_size, limit)
            print(f" {res['wall']:.1f}s wall, {res['sent_per_sec']:.1f} sent/s")
            results.append(res)

    # Qualitative output uses full-description spans - the most complete extraction.
    full_runs = [r for r in results if r["trunc"] == "full"]
    source = full_runs[0] if full_runs else results[0]
    combo_label = f"batch_size={source['batch_size']}, truncation={source['trunc']}"

    from jobscout.profiles import RuleBasedExtractor

    extractor = RuleBasedExtractor()
    job_rules = [extractor.extract_from_text(text)["skills"] for text in job_texts]

    cv_payload = None
    if cv_text is not None:
        print("\nCV pass (batch_size=16, full text)...")
        t0 = time.perf_counter()
        cv_spans, cv_sentences = ner.extract(cv_text, batch_size=16)
        cv_elapsed = time.perf_counter() - t0
        print(
            f"  {cv_elapsed:.2f}s for {cv_sentences} sentences "
            f"({cv_sentences / cv_elapsed:.1f} sent/s), {len(cv_spans)} spans"
        )
        cv_payload = (
            cv_text,
            cv_spans,
            extractor.extract_from_text(cv_text)["skills"],
        )

    total_ner, total_rules = write_report(
        jobs, source["spans"], job_rules, cv_payload, cv_status, combo_label
    )

    rss, rss_source = peak_rss_mb()

    print("\n" + "=" * 78)
    print(f"TIMING SUMMARY - {len(jobs)} longest job descriptions, CPU")
    print("=" * 78)
    header = (
        f"{'batch':>5} {'trunc':>7} {'wall(s)':>8} {'mean(s)':>8} {'p95(s)':>7} "
        f"{'sents':>6} {'sent/s':>7} {str(EXTRAPOLATE_TO) + 'jobs(min)':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['batch_size']:>5} {r['trunc']:>7} {r['wall']:>8.1f} "
            f"{r['mean']:>8.2f} {r['p95']:>7.2f} {r['sentences']:>6} "
            f"{r['sent_per_sec']:>7.1f} {r['extrapolated_min']:>14.1f}"
        )
    print("-" * len(header))
    print(f"model load time : {load_time:.1f}s (excluded from every row above)")
    if rss is None:
        print(f"peak RSS        : unavailable on this platform ({sys.platform})")
    else:
        print(f"peak RSS        : {rss:.0f} MB  [{rss_source}]")
    print(f"p95 is nearest-rank over {len(jobs)} per-job samples.")

    scope = f"{len(jobs)} jobs" + (" + CV" if cv_payload else "")
    print(
        f"\nTotal NER spans: {total_ner}  vs  total rule-based skills: {total_rules}  "
        f"(across the {scope}, deduplicated per document)"
    )
    print(f"\nQualitative deliverable: {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
