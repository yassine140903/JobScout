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

# Start the web UI
jobscout serve --port 8000
```

`jobscout profiles` lists the profiles you've ingested. Every command accepts
`--config path/to/config.yaml`.

## Configuration

All configuration lives in `config.yaml` at the project root.

```yaml
db_path: jobscout.db

model:
  name: intfloat/multilingual-e5-base

scoring:
  weights:
    skills: 0.60
    domain: 0.40

sources:
  - name: wttj
    adapter: wttj
    enabled: true
    keywords: []            # empty → filled from your profile
    max_pages: 3

  - name: eures
    adapter: eures
    enabled: true
    keywords: []
    locations: []           # empty → filled from your profile
    max_pages: 3

  - name: euraxess
    adapter: euraxess
    enabled: false          # opt-in, see below
    keyword: null
```

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

| Adapter | Source | Default |
| --- | --- | --- |
| `wttj` | Welcome to the Jungle — startup/tech jobs, mostly FR | enabled |
| `eures` | EURES — the EU public employment portal | enabled |
| `euraxess` | EURAXESS — academic and research positions | disabled |
| `generic_rss` | Any RSS, Atom, or JSON feed you point it at | — |

EURAXESS is disabled by default and opt-in. It has no public API, so the adapter
scrapes HTML: it is fragile, rate-limited with deliberate delays, and may break
whenever the site's theme changes. Enable it knowing that.

`generic_rss` is how you add a feed without writing code. Give it a URL and,
if the feed's field names differ from the defaults, a `field_map`:

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

## License

MIT License

Copyright (c) 2026 Yassine Ben JEMAA
