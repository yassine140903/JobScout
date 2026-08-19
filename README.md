# JobScout

Local-first, CV-driven job discovery tool.

## What it does

JobScout scrapes European job boards, embeds the postings with a multilingual
sentence-transformer model, and scores every posting against *your* CV. You get
a ranked list in a browser UI, with a per-job breakdown of why it scored the way
it did.

Everything runs on your machine: your CV, the SQLite database, and the embedding
model never leave it.

**JobScout does not auto-apply to anything.** It finds and ranks postings — the
applying is still up to you.

## Quick start

```bash
uv sync
uv run jobscout serve
```

(First sync downloads ~2 GB for PyTorch and the embedding model.)

**No API key is needed.** Every enabled source is a public endpoint, the
embedding model runs locally, and there is no account to create. Nothing in
`config.yaml` is a credential.

Open <http://localhost:8000>, upload your CV (PDF or DOCX), and click **Run**.

`jobscout serve` creates the database and loads sample postings on first run, so
you can see the UI working before configuring any real sources.

The first scoring run loads the embedding model and takes a few minutes.
Subsequent runs are fast.

## CLI usage

```bash
# Create the database and load sample postings
jobscout init

# Ingest a CV and build a profile from it
jobscout ingest cv.pdf --name industry-mle --locations "Paris, Berlin"

# Score all jobs in the DB against a profile
jobscout match industry-mle

# Fetch fresh postings from the sources in config.yaml
jobscout fetch
jobscout fetch --sources wttj,eures -v

# Remove stored postings the current language filter would now reject
jobscout prune            # dry run: reports what would go, deletes nothing
jobscout prune --apply    # actually delete

# Start the web UI
jobscout serve --port 8000
```

`jobscout profiles` lists the profiles you've ingested. Every command accepts
`--config path/to/config.yaml`.

`jobscout match` takes `--show-stretch` to include jobs filtered out for
demanding more years than you have.

`jobscout prune` exists because source filters apply at fetch time only: rows
stored before a filter was added, or before it was narrowed, stay in the
database and keep being scored. It reconciles what is stored with what the
config now says to keep. It is a dry run unless you pass `--apply`, it only
touches sources that declare a `languages` filter, it keeps rows whose language
was never detected, and it refuses to delete any posting referenced by the
evaluation set in `tests/eval/skills_gold.yaml`.

## Configuration

All configuration lives in `config.yaml` at the project root.

```yaml
db_path: jobscout.db

model:
  name: intfloat/multilingual-e5-base

profile:
  candidate_years: 0.75     # your years of experience; null = neutral

scoring:
  weights:
    skills: 0.60
    domain: 0.40
  seniority:
    gate_years: 2.0
    filter_on_inferred: false

sources:
  - name: wttj
    adapter: wttj
    enabled: true
    keywords: []            # empty → filled from your profile
    max_pages: 5
    fetch_details: true     # second request per posting for the real text
    detail_concurrency: 8

  - name: eures
    adapter: eures
    enabled: true
    keywords: []
    locations: []           # empty → filled from your profile
    max_pages: 3
    languages: [fr, en, de] # null = keep every language

  - name: euraxess
    adapter: euraxess
    enabled: false          # opt-in, see below
    keyword: null
```

**`profile.candidate_years`** is your years of professional experience, and it
is the authoritative figure. JobScout also parses date ranges out of your CV,
but that reading is advisory and never overwrites this value — `jobscout
profiles` shows you what it inferred. Set it to `null` to switch the seniority
facet off entirely: every posting then scores 1.0 on seniority and none is
filtered.

**`scoring.seniority`** governs how experience gaps are treated. The asymmetry
is deliberate: being under a posting's requirement decays the score steeply and,
past `gate_years`, removes the posting from the default view; being over it is
barely penalised and never filters. `filter_on_inferred` decides whether a
requirement *guessed from the job title* may filter — off by default, because
titles are inflated often enough that a wrong filter loses the job, while a
wrong penalty only costs rank position. A requirement stated by the source or
found in the posting's own prose may always filter.

**Scoring weights** control how the final score is composed. `skills` weights
the similarity between your skills and the posting; `domain` weights the
similarity between your field of work and the posting. They should sum to 1.0.
Seniority is applied as a multiplier on top.

**Sources** is a list of adapter configs. Each entry needs an `adapter` key
(which built-in adapter to use) and a `name` (how postings are labelled in the
DB). `enabled: false` skips it.

Leaving `keywords` or `locations` empty is the recommended default: JobScout
fills them in from your profile's domains, skills, and target locations at run
time. Set them explicitly only when you want to override that.

## Built-in sources

| Adapter | Source | Default | Status |
| --- | --- | --- | --- |
| `wttj` | Welcome to the Jungle — startup/tech jobs, mostly FR | enabled | stable, two-stage fetch |
| `eures` | EURES — the EU public employment portal | enabled | stable, language-filtered |
| `generic_rss` | Any RSS, Atom, or JSON feed you point it at | — | stable |
| `euraxess` | EURAXESS — academic and research positions | disabled | fragile, HTML scraping |

**`wttj` fetches in two stages.** Its Algolia search index carries no job
descriptions — only a short requirements blurb, null on about 15% of postings —
so the adapter makes one additional request per posting against a separate
public detail endpoint and joins that posting's `description`, `profile` and
`recruitment_process` fields into the stored text. This is what `fetch_details`
and `detail_concurrency` control. Failures are per-posting and non-fatal: a
posting that cannot be fetched falls back to the index blurb, a 404 marks the
posting delisted rather than failing the run, and a detail failure rate above
10% is logged as a warning that the endpoint may be throttling. Turn
`fetch_details` off and the run degrades to index-only instead of failing.

**`eures` is filtered by language.** It spans the entire EU, and the skill
vocabulary covers French, English and German only; postings outside those score
on an empty skill set, which ranks them on domain similarity alone. The default
`languages: [fr, en, de]` drops the rest at ingest. Set it to `null` to keep
every language. The filter applies to new fetches only — use `jobscout prune`
to reconcile rows stored before you set it.

EURAXESS is disabled by default and opt-in. It has no public API, so the adapter
scrapes HTML: it is fragile, rate-limited with deliberate delays, and may break
whenever the site's theme changes. Enable it knowing that.

## Pointing it at your own board

`generic_rss` adds a feed without writing any code. It reads RSS, Atom, and
JSON feeds. Give it a URL and, if the feed's field names differ from the
defaults, a `field_map` mapping JobScout's fields to the feed's own — dotted
paths work for nested values:

```yaml
  - name: my_uni_feed
    adapter: generic_rss
    enabled: true
    url: https://example.edu/jobs/feed.xml
    country: FR
    field_map:
      company: author.name
      description: content
```

## Writing a custom adapter

When a source needs real parsing logic, write an adapter: subclass
`SourceAdapter`, implement `fetch(source_config) -> list[RawPosting]`, register
it in `_get_adapters()`, and add an entry to `config.yaml`. Adapter failures are
isolated — one broken adapter won't take down a run. See
[ADAPTERS.md](ADAPTERS.md) for the full walkthrough.

## Stack

- Python 3.11+, managed with [uv](https://docs.astral.sh/uv/)
- SQLite for storage — one file, no server
- FastAPI + Jinja2 + HTMX for the UI (no build step, no JS framework)
- sentence-transformers with `intfloat/multilingual-e5-base` for embeddings

## Tests

```bash
uv run pytest tests/ -v
```

Extraction quality is measured separately against a hand-labelled gold set:

```bash
uv run python tests/eval/run_eval.py
```

It reports precision, recall and F1 for skill extraction over the postings
labelled in `tests/eval/skills_gold.yaml`. See the known limitations in
[jobscout-spec.md](jobscout-spec.md) for what those numbers do and do not
currently tell you.

## License

MIT License

Copyright (c) 2026 Yassine Ben JEMAA
