# Writing a source adapter

A source adapter turns one job board into a list of `RawPosting` objects.
Everything downstream — normalization, dedup, org-type classification,
insertion, scoring — is handled for you.

Before writing one, check whether `generic_rss` already covers your source: if
it publishes an RSS, Atom, or JSON feed, you can add it in `config.yaml` with no
code at all. See the README.

## The interface

```python
class SourceAdapter(ABC):
    """Base class for all source adapters."""

    name: str

    @abstractmethod
    def fetch(self, source_config: dict) -> list[RawPosting]:
        """Fetch postings from this source. May raise on network errors."""
        ...
```

`name` is the adapter's identifier — it lands in each posting's `source` column.
`source_config` is the matching entry from `config.yaml`, passed through as-is,
so any key you put there is available to your adapter.

## RawPosting

```python
@dataclass
class RawPosting:
    title: str                     # required — job title
    source: str                    # required — adapter name, e.g. "wttj"
    company: str | None = None     # hiring organization
    url: str | None = None         # link to the posting
    description: str | None = None # full text; what gets embedded
    location: str | None = None    # human-readable, e.g. "Paris"
    country: str | None = None     # ISO 3166-1 alpha-2, e.g. "FR"
    language: str | None = None    # posting language: en / fr / de
    seniority: str | None = None   # junior / mid / senior, if the source says
    posted_at: str | None = None   # ISO date string
    source_id: str | None = None   # the source's own ID, for same-source dedup
    raw_data: dict | None = None   # original payload, stored as JSON
```

Only `title` and `source` are required, but `description` is what the matcher
embeds — a posting without one will score poorly. Fill in as much as the source
gives you.

## A minimal adapter

```python
# src/jobscout/sources/acmejobs.py
"""AcmeJobs adapter — public JSON API."""

from __future__ import annotations

import logging

import httpx

from jobscout.sources import RawPosting, SourceAdapter

logger = logging.getLogger(__name__)

API_URL = "https://api.acmejobs.example/v1/postings"


class AcmeJobsAdapter(SourceAdapter):
    name = "acmejobs"

    def fetch(self, source_config: dict) -> list[RawPosting]:
        keywords = source_config.get("keywords", [])
        max_results = source_config.get("max_results", 100)

        params = {"q": " ".join(keywords), "limit": max_results}

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        postings = []
        for item in data.get("results", []):
            title = item.get("title")
            if not title:
                continue  # skip junk rather than raising
            postings.append(RawPosting(
                title=title,
                source=self.name,
                company=item.get("employer"),
                url=item.get("apply_url"),
                description=item.get("body"),
                location=item.get("city"),
                country=item.get("country_code"),
                posted_at=item.get("published_at"),
                source_id=str(item.get("id")),
                raw_data=item,
            ))

        logger.info("%s: fetched %d postings", self.name, len(postings))
        return postings
```

Naming `keywords` and `locations` in your config schema is worth doing: when
those lists are left empty, `enrich_config_from_profile()` fills them from the
user's profile automatically.

## Registering it

Add the class to the registry in `_get_adapters()` in `sources/__init__.py`:

```python
from jobscout.sources.acmejobs import AcmeJobsAdapter

registry: dict[str, type[SourceAdapter]] = {
    "wttj": WTTJAdapter,
    "eures": EURESAdapter,
    "euraxess": EURAXESSAdapter,
    "generic_rss": GenericRSSAdapter,
    "acmejobs": AcmeJobsAdapter,
}
```

Then add a source entry to `config.yaml`:

```yaml
  - name: acmejobs
    adapter: acmejobs
    enabled: true
    keywords: []
    max_results: 50
```

## Testing

Test the parsing logic against a mocked HTTP response — no network in tests:

```python
from unittest.mock import patch, MagicMock


def test_acmejobs_parses_postings():
    from jobscout.sources.acmejobs import AcmeJobsAdapter

    fake = MagicMock()
    fake.json.return_value = {
        "results": [{"id": 1, "title": "ML Engineer", "employer": "Acme"}]
    }
    fake.raise_for_status.return_value = None

    with patch("httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.return_value = fake
        postings = AcmeJobsAdapter().fetch({"keywords": ["ml"]})

    assert len(postings) == 1
    assert postings[0].title == "ML Engineer"
    assert postings[0].source == "acmejobs"
```

To test how your adapter behaves inside a full run, use the `FakeAdapter`
pattern in `tests/test_m4.py` — a stub `SourceAdapter` returning canned postings
(or raising a canned exception), which lets you exercise `run_fetch()` end to
end without touching the network.

## Failure isolation

`run_fetch()` wraps each adapter in its own try/except. If yours raises, the
error is logged and recorded on the run row, and the remaining adapters still
run — one broken source never kills the whole fetch. Network errors
(`ConnectError`, timeouts, and friends) get one automatic retry before that.

So prefer raising a clear exception over returning half-parsed garbage, and skip
individual malformed items rather than aborting the batch.
