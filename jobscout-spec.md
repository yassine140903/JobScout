# JobScout — Project Spec & Milestones

> Working document for a local-first, CV-driven job discovery tool.
> Status: pre-M1. Name is provisional.

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
- **Dedup layer**: URL hash + fuzzy match on (title, company). Jobs repost,
  cross-post, and get reworded — without this the list is noise.
- Per-adapter failure isolation: one dead scraper must not kill the run
- Sources declared in `config.yaml` so a fork can swap them entirely

Initial three adapters:

1. **EURAXESS** — RSS, trivial to parse, high signal for research positions
2. **Welcome to the Jungle** — strong FR/EU coverage
3. **Generic RSS/JSON adapter** — configurable, lets anyone point at their own
   board without writing code

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

---

## 7. Working method

Per milestone: discuss and surface tradeoffs → produce a scoped prompt for
Claude Code → joint review of the resulting code. One milestone at a time.
