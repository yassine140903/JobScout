"""Generic RSS/Atom/JSON feed adapter — configurable via config.yaml."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import feedparser
import httpx

from jobscout.sources import RawPosting, SourceAdapter

logger = logging.getLogger(__name__)

# Default mapping: feedparser entry keys → RawPosting fields
DEFAULT_FIELD_MAP = {
    "title": "title",
    "company": "author",
    "url": "link",
    "description": "summary",
    "posted_at": "published",
    "source_id": "id",
}


class GenericRSSAdapter(SourceAdapter):
    name = "generic_rss"

    def fetch(self, source_config: dict) -> list[RawPosting]:
        url = source_config.get("url")
        if not url:
            logger.warning("generic_rss: no URL configured, skipping")
            return []

        feed_name = source_config.get("name", "generic_rss")
        field_map = {**DEFAULT_FIELD_MAP, **source_config.get("field_map", {})}
        max_entries = source_config.get("max_entries", 100)
        default_country = source_config.get("country")
        default_language = source_config.get("language")
        default_location = source_config.get("location")

        raw_feed = self._fetch_feed(url)
        feed = feedparser.parse(raw_feed)

        if feed.bozo and not feed.entries:
            logger.error(
                "generic_rss [%s]: feed parse failed — %s",
                feed_name, feed.bozo_exception,
            )
            return []

        if feed.bozo:
            logger.warning(
                "generic_rss [%s]: feed has errors but %d entries parsed — %s",
                feed_name, len(feed.entries), feed.bozo_exception,
            )

        postings: list[RawPosting] = []
        for entry in feed.entries[:max_entries]:
            posting = self._entry_to_posting(
                entry, field_map, feed_name,
                default_country, default_language, default_location,
            )
            if posting:
                postings.append(posting)

        logger.info("generic_rss [%s]: %d postings from %s", feed_name, len(postings), url)
        return postings

    @staticmethod
    def _fetch_feed(url: str) -> bytes:
        """Fetch raw feed content via httpx (feedparser's built-in fetcher is limited)."""
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={
                    "Accept": "application/rss+xml, application/atom+xml, application/json, text/xml, */*",
                    "User-Agent": "Mozilla/5.0 (compatible; JobScout/1.0; feed reader)",
                },
            )
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _entry_to_posting(
        entry: dict[str, Any],
        field_map: dict[str, str],
        source_name: str,
        default_country: str | None,
        default_language: str | None,
        default_location: str | None,
    ) -> RawPosting | None:
        """Convert a feedparser entry to a RawPosting using the field map."""

        def get_field(our_field: str) -> str | None:
            feed_key = field_map.get(our_field)
            if not feed_key:
                return None
            value = _resolve_nested(entry, feed_key)
            if value is None:
                return None
            return str(value).strip() or None

        title = get_field("title")
        if not title:
            return None

        url = get_field("url")

        # Build source_id: prefer feed-provided id, fall back to URL hash
        source_id = get_field("source_id")
        if not source_id and url:
            source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        # Description: try mapped field, then fall back to content
        description = get_field("description")
        if not description:
            content_list = entry.get("content", [])
            if content_list and isinstance(content_list, list):
                description = content_list[0].get("value")

        # Strip HTML tags from description if present
        if description:
            description = _strip_html(description)

        return RawPosting(
            title=title,
            source=source_name,
            company=get_field("company"),
            url=url,
            description=description,
            location=get_field("location") or default_location,
            country=get_field("country") or default_country,
            language=get_field("language") or default_language,
            seniority=get_field("seniority"),
            posted_at=get_field("posted_at"),
            source_id=source_id,
            raw_data=dict(entry),
        )


def _resolve_nested(data: dict, key: str) -> Any:
    """Resolve a potentially dot-separated key like 'author.name'."""
    parts = key.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _strip_html(text: str) -> str:
    """Strip markup from a feed description, preserving block structure.

    Delegates to the shared cleaner. The previous regex flattened everything
    to a single line, which destroyed the structure that lets a truncated
    description keep its requirements section.
    """
    from jobscout.textclean import clean_description

    return clean_description(text)