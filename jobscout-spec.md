# JobScout — Project Spec & Milestones

> Working document for a local-first, CV-driven job discovery tool.
> Status: M1-M7 complete. See "Current status" below.

---

## 0. Current status

M1-M7 are complete. The tool ingests a CV, fetches from WTTJ, EURES and any
generic RSS/JSON feed, scores every posting, and serves a ranked list.

M7 was an extraction-quality milestone rather than a feature milestone. What it
changed:

- **Vocabulary composition.** The recall gap was concept and practice terms
  (`ci/cd`, `iac`, `feature store`, `model serving`), not product names. Added
  23 such terms.
- **REST word order.** French inverts the noun phrase ("API REST") and English
  writes "RESTful" as often as "REST API". Aliases now cover the word orders;
  separators and plurals fall out of the pattern generator. Corpus coverage went
  from 140 to 199 of the 199 documents that mention REST in any form.
- **Case handling.** Skill patterns are compiled with `re.IGNORECASE` rather
  than relying on the caller to lowercase its input, a precondition that was
  invisible at the call site. `R` stays case-sensitive on its own branch.
- **Position type.** `classify_position_type` matched on substrings, so
  "mission" inside ordinary prose read as freelance. It is word-bounded now,
  and title markers are checked before description markers.
- **EURES language filter.** `sources.eures.languages`, default `[fr, en, de]`,
  applied at ingest. Dutch, Italian and Swedish rows were overwhelmingly
  non-technical postings that scored on an empty skill set.
- **`jobscout prune`.** Source filters apply at fetch time only, so rows stored
  before a filter existed keep being scored. `prune` reconciles stored rows with
  the current config. Dry run by default; never deletes a posting the gold set
  references.
- **Seniority provenance.** The requirement is resolved api -> description ->
  title -> none, and which layer produced it is recorded and shown. Only a
  stated requirement may filter a posting out of view. An API value of exactly
  0 is treated as "not stated", because WTTJ uses 0 for both.

### Known limitations

These are measured, not suspected. None is fixed as of the M7 close-out.

- **Score banding.** Final scores compress into roughly 0.85-0.90. The ranking
  within that band is meaningful but the absolute numbers are not, and small
  score differences should not be read as large quality differences.
- **`go` and `vue` short-token false positives.** `go` matches 178 documents,
  113 of which contain no genuine reference - mostly the recruiting boilerplate
  "so go ahead and apply". `vue` matches 117 documents, 45 of them the French
  "point de vue" / "en vue de". Both are correctly word-bounded; the terms are
  simply ordinary words as well as technologies.
- **`c#` and `c++` match nothing.** Their patterns end in ``, and `#` and `+`
  are not word characters, so the boundary can never be satisfied. Two silently
  dead vocabulary entries.
- **The gold set does not measure precision.** 19 of its 39 entries are
  labelled, and all 19 sit in the `rich_skills` stratum. The 12 `zero_skills`
  and 4 `non_technical` entries - the ones that would catch invented skills -
  are still unlabelled. Reported precision of 1.0 reflects that gap, not the
  extractor.
- **Cross-aggregator duplicates.** Dedup hashes `(title, company)`. Aggregators
  such as Forums Talents Handicap republish a posting under their own company
  name, so the hash legitimately differs and the duplicate survives. A corpus
  scan on description similarity found 8 cross-source near-duplicate pairs at
  Jaccard >= 0.70; only half share a company string. Detecting these needs
  content similarity, not a hash.

---

## 1. What this is

A local tool you point at your CV. It scrapes job boards, embeds each posting,
scores it against your profile, and shows you a ranked, filterable list in a
browser.

**It does not auto-apply.** It does not do outreach. It sources and ranks.

## 2. Constraints (these drive every decision below)

| Constraint | Consequence |
|---|---|
| Real daily-use tool, not a portfolio piece | Optimize for signal quality and durability, not architectural showcase |
| Shared via GitHub, run locally by others | Clone → one command → working. Every extra dep is a user who never runs it |
| No deployment | No Docker requirement, no managed services, no cloud DB |
| "Working, not perfect" | Cut anything not on the path to ranked jobs in a browser |
| Multi-country search (FR/DE/EN postings) | Multilingual embeddings are mandatory, not optional |

## 3. Locked stack

- **Python 3.11+**, dependencies via `uv`
- **SQLite** — tables: `profiles`, `jobs`, `matches`, `runs`. Zero setup.
- **FastAPI + Jinja2 + HTMX** — single process, no Node toolchain, no build step
- **sentence-transformers**, model `intfloat/multilingual-e5-base`
  (alternative: `BAAI/bge-m3`). Embeddings stored as SQLite BLOBs.
- **numpy** for similarity. No vector DB — at 5–20k jobs a dot product beats
  Qdrant and adds zero deps.
- **httpx + selectolax** for scraping
- **Optional LLM** (Anthropic/OpenAI key in `.env`) for CV parsing and match
  rationale. Rule-based fallback when absent — the tool must work with no key.

### Run model

On-demand only. No scheduler, no Celery, no Redis.

A scrape run takes 1–3 minutes, so it cannot block the request:
background thread writes progress to the `runs` table, HTMX polls
`/runs/{id}/status` for a progress bar.

---

## 4. Milestones

Each milestone is a self-contained unit of work. M1–M3 are fully offline and
testable against fixtures — the matching engine is proven working before any
fragile scraper code exists. That ordering is deliberate.

### M1 — Core + storage

Project skeleton, SQLite schema, config loading, CLI entry point.

- `uv`-managed project, package layout, `jobscout` console script
- Schema for `profiles`, `jobs`, `matches`, `runs`
- `config.yaml` loading (sources, scoring weights, model name)
- **Fixture set of ~20 realistic fake postings** (mixed EN/FR, mixed
  seniority, mixed relevance) — everything downstream is testable without
  network access

No scraping. No UI. No embeddings.

**Done when:** `jobscout init` creates the DB and loads fixtures; schema is
queryable.

---

### M2 — Profiles + CV ingestion

- PDF/DOCX → raw text extraction
- Text → structured profile: skills, domains, seniority, languages,
  target locations
- Rule-based extractor as default path
- LLM enrichment behind a flag, gracefully skipped without a key
- **Multiple named profiles per install** (e.g. `research-phd`,
  `industry-mle`) — same CV can produce different profiles that score jobs
  differently

**Done when:** a real CV produces a stored, inspectable profile; two profiles
coexist independently.

---

### M3 — Embedding + matching engine

- Faceted embedding of both profile and job (skills / domain / seniority /
  location rather than one blob)
- Cosine similarity per facet, weighted combination into a final score
- **Explainability payload** — which facets drove the score, so the UI can
  say "matched on: MLOps, computer vision, Python"
- Tuning weights exposed in config (and later in the UI — ~30 lines, worth it
  for comparing "research fit" vs "industry fit")

Tested entirely against M1 fixtures. Zero network dependency.

**Done when:** fixtures rank in a defensible order and scores are explainable.

---

### M4 — Source adapters

- `SourceAdapter` ABC: `fetch() -> list[RawPosting]`
- Normalization to a canonical `Job` model
- **Dedup layer**: URL hash + hash of (title, company). Jobs repost,
  cross-post, and get reworded — without this the list is noise.
  *Shipped as an exact hash, not a fuzzy match* — see the cross-aggregator
  duplicate limitation in §0.
- Per-adapter failure isolation: one dead scraper must not kill the run
- Sources declared in `config.yaml` so a fork can swap them entirely

Initial three adapters (as planned):

1. **EURAXESS** — RSS, trivial to parse, high signal for research positions
2. **Welcome to the Jungle** — strong FR/EU coverage
3. **Generic RSS/JSON adapter** — configurable, lets anyone point at their own
   board without writing code

**As shipped**, four adapters exist and the defaults differ from the plan:

| Adapter | Default | Note |
|---|---|---|
| `wttj` | enabled | Two-stage: the search index carries no descriptions, so a second request per posting fetches the real text |
| `eures` | enabled | Added during M4; language-filtered to `fr/en/de` since M7 |
| `generic_rss` | — | As planned |
| `euraxess` | disabled | No public API, so it scrapes HTML; fragile and opt-in |

EURAXESS was planned as the easy first adapter and turned out to be the fragile
one. EURES took its place as a default source.

Deliberately excluded for now: LinkedIn, Glassdoor. They actively fight
scrapers and will break constantly. Not worth the maintenance tax.

**Done when:** a real run pulls, normalizes, dedups, and scores live postings.

---

### M5 — Web UI

- Run trigger + live progress (HTMX polling `runs` status)
- Results table: score, title, company, source, date, location
- Filters: score threshold, source, location, profile
- Job detail pane with full description and score breakdown
- Status flags per job: saved / applied / dismissed

**Done when:** the full loop — pick profile, run, browse, flag — works in a
browser with no CLI involvement.

---

### M6 — Polish & shareability

- README with one-command setup
- Adapter-writing guide (the thing that makes forks useful)
- Error handling and clear messaging for dead scrapers
- Sensible defaults so first run works with zero configuration

---

## 5. Open decisions

- [ ] **Name.** `JobScout` is a placeholder. It becomes the package name —
      decide before M1 to avoid a refactor.
- [ ] **Scoring weights in UI.** Config-only for M3, promoted to UI in M5?
      Leaning yes.
- [ ] **Adapter set beyond the first three.** Driven by actual search targets
      (individual lab career pages are high-signal but each is bespoke).

## 6. Explicitly out of scope

- Auto-apply
- Cold outreach / email generation
- Deployment, hosting, multi-user accounts
- Scheduler / background daemon
- Vector database
- LinkedIn and Glassdoor scraping


