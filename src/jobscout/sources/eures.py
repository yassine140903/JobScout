"""EURES adapter — via public REST API (europa.eu)."""

from __future__ import annotations

import ast
import json
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
# Languages whose postings the skill vocabulary can actually read. The rest
# of the EU is still fetched and simply not kept - EURES has no language
# facet we can filter on server-side, so this is an ingest filter.
DEFAULT_LANGUAGES: tuple[str, ...] = ("fr", "en", "de")


class EURESAdapter(SourceAdapter):
    name = "eures"

    def fetch(self, source_config: dict) -> list[RawPosting]:
        keywords = source_config.get("keywords", [])
        if not keywords:
            logger.warning("eures: no keywords configured, skipping")
            return []

        # EURES spans the whole EU, and the long tail of its languages is not
        # covered by the skill vocabulary - those postings score on an empty
        # skill set, which is worse than not having them. Set to null or [] to
        # keep every language.
        languages = source_config.get("languages", DEFAULT_LANGUAGES)
        languages = {lang.lower() for lang in languages} if languages else None

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

        if languages is not None:
            kept = [
                p for p in all_postings
                if (p.language or "").lower()[:2] in languages
            ]
            dropped = len(all_postings) - len(kept)
            if dropped:
                logger.info(
                    "eures: dropped %d of %d postings outside languages %s",
                    dropped, len(all_postings), sorted(languages),
                )
            all_postings = kept

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
                try:
                    posting = self._item_to_posting(item)
                except (AttributeError, TypeError, KeyError) as exc:
                    logger.debug("eures: skipping malformed item — %s", exc)
                    continue
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
        if isinstance(location_map, dict):
            for country_code, regions in location_map.items():
                countries.add(country_code.upper())
                if isinstance(regions, list):
                    for region in regions:
                        if isinstance(region, dict):
                            name = region.get("label") or region.get("nuts3Label")
                            if name:
                                locations.append(name)
                        elif isinstance(region, str):
                            locations.append(region)

        location_str = ", ".join(locations) if locations else None
        country_str = list(countries)[0] if len(countries) == 1 else (
            ",".join(sorted(countries)) if countries else None
        )

        # Employer — an object, not a string; only its name belongs in `company`
        employer = employer_name(item.get("employer") or item.get("employerName"))

        # Description from snippet/summary
        description = item.get("description") or item.get("snippet")

        # Dates
        posted = item.get("publicationDate") or item.get("modificationDate")

        # Experience level
        experience = item.get("requiredExperienceMonths")
        seniority = _map_experience(experience)

        # Language
        available = item.get("availableLanguages")
        language = item.get("language")
        if not language and available and isinstance(available, list):
            language = available[0] if available else "en"
        language = language or "en"

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


def employer_name(employer: Any) -> str | None:
    """Pull a plain company name out of whatever EURES puts in `employer`.

    EURES returns an object — {"name": ..., "legalID": ..., "sectorCodes": ...}
    — and stringifying the whole thing put dict reprs in `company`, which is
    the dedup key. Only the name is wanted.
    """
    if employer is None:
        return None
    if isinstance(employer, dict):
        name = employer.get("name") or employer.get("employerName")
        employer = name if name is not None else None
    if employer is None:
        return None
    name = str(employer).strip()
    return name or None


def repair_company_value(stored: Any) -> tuple[str | None, str]:
    """Repair one already-stored `company` value. Returns (value, outcome).

    Outcome is 'clean' (nothing to do), 'repaired', or 'unparseable' — the last
    leaves the value untouched rather than guessing at it.
    """
    if stored is None:
        return None, "clean"
    text = str(stored).strip()
    if not text.startswith("{"):
        return stored, "clean"

    # Values were written with str(dict), so they are Python reprs (None, not
    # null) — but try JSON too in case a source ever wrote real JSON.
    for parse in (ast.literal_eval, json.loads):
        try:
            obj = parse(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        name = employer_name(obj)
        if name:
            return name, "repaired"
        return stored, "unparseable"   # parsed, but carried no usable name
    return stored, "unparseable"


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