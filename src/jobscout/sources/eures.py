"""EURES adapter — via public REST API (europa.eu)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from jobscout.sources import RawPosting, SourceAdapter

logger = logging.getLogger(__name__)

BASE_URL = "https://europa.eu/eures/api"
SEARCH_ENDPOINT = f"{BASE_URL}/jv-searchengine/public/jv-search/search"
DETAIL_ENDPOINT = f"{BASE_URL}/jv-searchengine/public/jv/id"
DEFAULT_RESULTS_PER_PAGE = 50
DEFAULT_MAX_PAGES = 5


class EURESAdapter(SourceAdapter):
    name = "eures"

    def fetch(self, source_config: dict) -> list[RawPosting]:
        keywords = source_config.get("keywords", [])
        if not keywords:
            logger.warning("eures: no keywords configured, skipping")
            return []

        locations = source_config.get("locations", [])  # NUTS codes, e.g. ["fr", "de"]
        results_per_page = source_config.get("results_per_page", DEFAULT_RESULTS_PER_PAGE)
        max_pages = source_config.get("max_pages", DEFAULT_MAX_PAGES)
        fetch_details = source_config.get("fetch_details", False)

        all_postings: list[RawPosting] = []
        seen_ids: set[str] = set()

        with httpx.Client(timeout=30) as client:
            for keyword in keywords:
                postings = self._search_keyword(
                    client, keyword, locations,
                    results_per_page, max_pages, fetch_details,
                )
                for p in postings:
                    key = p.source_id or p.url or p.title
                    if key not in seen_ids:
                        seen_ids.add(key)
                        all_postings.append(p)

        logger.info("eures: %d postings across %d keywords", len(all_postings), len(keywords))
        return all_postings

    def _search_keyword(
        self,
        client: httpx.Client,
        keyword: str,
        locations: list[str],
        results_per_page: int,
        max_pages: int,
        fetch_details: bool,
    ) -> list[RawPosting]:
        postings: list[RawPosting] = []
        page = 1  # EURES uses 1-indexed pages

        while page <= max_pages:
            body = _build_search_body(keyword, locations, page, results_per_page)
            resp = client.post(
                SEARCH_ENDPOINT,
                json=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("jvs", [])
            if not items:
                break

            for item in items:
                posting = self._item_to_posting(item)
                if posting:
                    # Optionally fetch full description from detail endpoint
                    if fetch_details and posting.source_id:
                        posting = self._enrich(client, posting)
                    postings.append(posting)

            total = data.get("numberResults", 0)
            if page * results_per_page >= total:
                break
            page += 1

        logger.debug("eures: %d postings for keyword %r", len(postings), keyword)
        return postings

    @staticmethod
    def _item_to_posting(item: dict[str, Any]) -> RawPosting | None:
        """Convert a EURES search result item to a RawPosting."""
        title = item.get("title")
        if not title:
            return None

        jv_id = item.get("id", "")
        # EURES detail page URL uses the encoded ID
        url = None
        if jv_id:
            url = f"https://europa.eu/eures/portal/jv-se/jv-details/{jv_id}"

        # Location from locationMap
        location_map = item.get("locationMap") or {}
        locations = []
        countries = set()
        for country_code, regions in location_map.items():
            countries.add(country_code.upper())
            if isinstance(regions, list):
                for region in regions:
                    name = region.get("label") or region.get("nuts3Label")
                    if name:
                        locations.append(name)

        location_str = ", ".join(locations) if locations else None
        country_str = list(countries)[0] if len(countries) == 1 else (
            ",".join(sorted(countries)) if countries else None
        )

        # Employer
        employer = item.get("employer") or item.get("employerName")

        # Description from snippet/summary
        description = item.get("description") or item.get("snippet")

        # Dates
        posted = item.get("publicationDate") or item.get("modificationDate")

        # Experience level
        experience = item.get("requiredExperienceMonths")
        seniority = _map_experience(experience)

        # Language
        language = item.get("language") or item.get("availableLanguages", ["en"])[0] if item.get("availableLanguages") else "en"

        return RawPosting(
            title=title,
            source="eures",
            company=employer,
            url=url,
            description=description,
            location=location_str,
            country=country_str,
            language=language,
            seniority=seniority,
            posted_at=posted,
            source_id=str(jv_id),
            raw_data=item,
        )

    def _enrich(self, client: httpx.Client, posting: RawPosting) -> RawPosting:
        """Fetch full job detail and merge in the description."""
        try:
            resp = client.get(
                f"{DETAIL_ENDPOINT}/{posting.source_id}",
                params={"requestLang": "en"},
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return posting
            detail = resp.json()

            # Full description from detail endpoint
            body = detail.get("body") or detail.get("description")
            if body:
                posting.description = body

            # Additional fields that might be richer in detail
            if not posting.company:
                posting.company = detail.get("employer") or detail.get("employerName")

        except httpx.HTTPError as exc:
            logger.debug("eures: detail fetch failed for %s — %s", posting.source_id, exc)

        return posting


def _build_search_body(
    keyword: str,
    locations: list[str],
    page: int,
    results_per_page: int,
) -> dict[str, Any]:
    """Build the EURES search API request body."""
    return {
        "resultsPerPage": results_per_page,
        "page": page,
        "sortSearch": "MOST_RECENT",
        "keywords": [
            {"keyword": keyword, "specificSearchCode": "EVERYWHERE"},
        ],
        "publicationPeriod": None,
        "occupationUris": [],
        "skillUris": [],
        "requiredExperienceCodes": [],
        "positionScheduleCodes": [],
        "sectorCodes": [],
        "educationAndQualificationLevelCodes": [],
        "positionOfferingCodes": [],
        "locationCodes": locations,
        "euresFlagCodes": [],
        "otherBenefitsCodes": [],
        "requiredLanguages": [],
        "minNumberPost": None,
        "sessionId": "jobscout-session",
        "requestLanguage": "en",
    }


def _map_experience(months: Any) -> str | None:
    """Map experience in months to seniority level."""
    if months is None:
        return None
    try:
        m = int(months)
    except (ValueError, TypeError):
        return None
    if m <= 6:
        return "intern"
    if m <= 24:
        return "junior"
    if m <= 60:
        return "mid"
    if m <= 120:
        return "senior"
    return "lead"