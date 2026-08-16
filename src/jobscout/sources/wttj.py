"""Welcome to the Jungle adapter — via public Algolia API."""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any

import httpx

from jobscout.sources import RawPosting, SourceAdapter

logger = logging.getLogger(__name__)

# Public search-only credentials (shipped in the WTTJ SPA bundle to every visitor)
ALGOLIA_URL = (
    "https://csekhvms53-dsn.algolia.net/1/indexes/*/queries"
    "?x-algolia-agent=Algolia+for+JavaScript+(4.20.0);+Browser"
)
ALGOLIA_APP_ID = "CSEKHVMS53"
ALGOLIA_API_KEY = "4bd8f6215d0cc52b26430765769e65a0"
JOB_INDEX = "wk_cms_jobs_production"
WTTJ_SITE = "https://www.welcometothejungle.com"
PAGINATION_CEILING = 1000  # Algolia hard cap on reachable results
DEFAULT_HITS_PER_PAGE = 100
DEFAULT_MAX_PAGES = 10


HEADERS = {
    "x-algolia-application-id": ALGOLIA_APP_ID,
    "x-algolia-api-key": ALGOLIA_API_KEY,
    "content-type": "application/x-www-form-urlencoded",  # Algolia CORS quirk
    "accept": "*/*",
    "origin": WTTJ_SITE,
    "referer": WTTJ_SITE + "/",
}


class WTTJAdapter(SourceAdapter):
    name = "wttj"

    def fetch(self, source_config: dict) -> list[RawPosting]:
        keywords = source_config.get("keywords", [])
        if not keywords:
            logger.warning("wttj: no keywords configured, skipping")
            return []

        hits_per_page = source_config.get("hits_per_page", DEFAULT_HITS_PER_PAGE)
        max_pages = source_config.get("max_pages", DEFAULT_MAX_PAGES)
        filters = source_config.get("filters")  # e.g. "offices.country_code:FR"

        all_postings: list[RawPosting] = []
        seen_ids: set[str] = set()

        with httpx.Client(timeout=30) as client:
            for keyword in keywords:
                postings = self._search_keyword(
                    client, keyword, hits_per_page, max_pages, filters,
                )
                # Deduplicate across keywords within this adapter
                for p in postings:
                    key = p.source_id or p.url or p.title
                    if key not in seen_ids:
                        seen_ids.add(key)
                        all_postings.append(p)

        logger.info("wttj: %d postings across %d keywords", len(all_postings), len(keywords))
        return all_postings

    def _search_keyword(
        self,
        client: httpx.Client,
        keyword: str,
        hits_per_page: int,
        max_pages: int,
        filters: str | None,
    ) -> list[RawPosting]:
        postings: list[RawPosting] = []
        page = 0

        while page < max_pages:
            # Respect Algolia's pagination ceiling
            if page * hits_per_page >= PAGINATION_CEILING:
                logger.info(
                    "wttj: hit pagination ceiling for %r at page %d", keyword, page,
                )
                break

            result = self._algolia_search(
                client, keyword, page, hits_per_page, filters,
            )
            hits = result.get("hits", [])
            if not hits:
                break

            for hit in hits:
                posting = self._hit_to_posting(hit)
                if posting:
                    postings.append(posting)

            nb_pages = result.get("nbPages", 0)
            if page + 1 >= nb_pages:
                break
            page += 1

        logger.debug("wttj: %d postings for keyword %r", len(postings), keyword)
        return postings

    def _algolia_search(
        self,
        client: httpx.Client,
        query: str,
        page: int,
        hits_per_page: int,
        filters: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": query,
            "hitsPerPage": hits_per_page,
            "page": page,
        }
        if filters:
            params["filters"] = filters

        body = {
            "requests": [
                {
                    "indexName": JOB_INDEX,
                    "params": urllib.parse.urlencode(params, safe=":,/"),
                }
            ]
        }
        resp = client.post(
            ALGOLIA_URL, headers=HEADERS, content=json.dumps(body),
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else {}

    @staticmethod
    def _hit_to_posting(hit: dict[str, Any]) -> RawPosting | None:
        title = hit.get("name")
        if not title:
            return None

        org = hit.get("organization") or {}
        org_slug = org.get("slug", "")
        job_slug = hit.get("slug", "")
        office = hit.get("office") or {}

        url = None
        if org_slug and job_slug:
            url = f"{WTTJ_SITE}/en/companies/{org_slug}/jobs/{job_slug}"

        # Build a plain-text description from available fields
        description_parts = []
        # Profile text (what they're looking for)
        profile = hit.get("profile")
        if isinstance(profile, dict):
            description_parts.append(profile.get("en") or profile.get("fr") or "")
        elif isinstance(profile, str):
            description_parts.append(profile)
        # Recruitment process
        recruitment = hit.get("recruitment_process")
        if isinstance(recruitment, dict):
            description_parts.append(recruitment.get("en") or recruitment.get("fr") or "")
        elif isinstance(recruitment, str):
            description_parts.append(recruitment)
        description = "\n\n".join(p for p in description_parts if p.strip())

        # Contract type
        contract = (hit.get("contract_type_names") or {}).get("en") or hit.get("contract_type")
        seniority = hit.get("experience_level") or hit.get("experience_level_minimum")

        return RawPosting(
            title=title,
            source="wttj",
            company=org.get("name"),
            url=url,
            description=description or None,
            location=office.get("city"),
            country=office.get("country_code") or office.get("country"),
            language=hit.get("language") or "en",
            seniority=_map_seniority(seniority),
            posted_at=hit.get("published_at"),
            source_id=str(hit.get("objectID", "")),
            raw_data=hit,
        )


def _map_seniority(level: Any) -> str | None:
    """Map WTTJ experience levels to our seniority terms."""
    if not level:
        return None
    mapping = {
        "LESS_THAN_6_MONTHS": "intern",
        "6_MONTHS_TO_1_YEAR": "junior",
        "1_TO_2_YEARS": "junior",
        "2_TO_3_YEARS": "mid",
        "3_TO_5_YEARS": "mid",
        "5_TO_10_YEARS": "senior",
        "MORE_THAN_10_YEARS": "lead",
    }
    return mapping.get(str(level), str(level).lower())