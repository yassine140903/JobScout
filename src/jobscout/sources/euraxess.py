"""EURAXESS adapter — HTML scraping (opt-in, experimental).

Note: EURAXESS (euraxess.ec.europa.eu) is a Drupal site with no public API.
This adapter scrapes search result pages with selectolax. It is:
  - Fragile: any Drupal theme update may break the selectors.
  - Marked 'enabled: false' by default in config.
  - Rate-limited with polite delays between requests.

Use responsibly. EURAXESS's robots.txt restricts some automated access.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx
from selectolax.parser import HTMLParser

from jobscout.sources import RawPosting, SourceAdapter

logger = logging.getLogger(__name__)

BASE_URL = "https://euraxess.ec.europa.eu"
SEARCH_URL = f"{BASE_URL}/jobs/search"
DEFAULT_MAX_PAGES = 3
DEFAULT_DELAY = 1.5  # seconds between requests — be polite


class EURAXESSAdapter(SourceAdapter):
    name = "euraxess"

    def fetch(self, source_config: dict) -> list[RawPosting]:
        max_pages = source_config.get("max_pages", DEFAULT_MAX_PAGES)
        delay = source_config.get("delay", DEFAULT_DELAY)
        params = _build_search_params(source_config)

        all_postings: list[RawPosting] = []

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            for page_num in range(max_pages):
                params["page"] = str(page_num) if page_num > 0 else None
                # Remove None values
                query = {k: v for k, v in params.items() if v is not None}

                logger.debug("euraxess: fetching page %d", page_num + 1)
                resp = client.get(
                    SEARCH_URL,
                    params=query,
                    headers=_request_headers(),
                )
                resp.raise_for_status()

                postings, has_next = self._parse_search_page(resp.text)
                all_postings.extend(postings)

                logger.debug(
                    "euraxess: page %d → %d postings (has_next=%s)",
                    page_num + 1, len(postings), has_next,
                )

                if not has_next or not postings:
                    break
                if page_num < max_pages - 1:
                    time.sleep(delay)

        logger.info("euraxess: %d postings total", len(all_postings))
        return all_postings

    def _parse_search_page(self, html: str) -> tuple[list[RawPosting], bool]:
        """Parse a EURAXESS search results page. Returns (postings, has_next)."""
        tree = HTMLParser(html)
        postings: list[RawPosting] = []

        # Each result is a list item in the search results
        # The structure uses <article> or list items with job details
        result_nodes = tree.css("div.views-row, li.views-row, article.node--type-job-offer")

        # Fallback: try to find result blocks by the heading links
        if not result_nodes:
            result_nodes = _find_result_blocks(tree)

        for node in result_nodes:
            posting = self._parse_result_node(node)
            if posting:
                postings.append(posting)

        # Check for "Next" pagination link
        has_next = bool(tree.css("a[rel='next'], li.pager__item--next a"))

        return postings, has_next

    def _parse_result_node(self, node: Any) -> RawPosting | None:
        """Parse a single search result node into a RawPosting."""
        # Title and URL from the heading link
        title_link = node.css_first("h3 a, h2 a, .field--name-node-title a")
        if not title_link:
            return None

        title = title_link.text(strip=True)
        if not title:
            return None

        href = title_link.attributes.get("href", "")
        url = f"{BASE_URL}{href}" if href.startswith("/") else href

        # Extract job ID from URL (/jobs/452595)
        source_id = None
        id_match = re.search(r"/jobs/(\d+)", href)
        if id_match:
            source_id = id_match.group(1)

        # Organization/company
        company = None
        org_link = node.css_first(
            "a[href*='/partnering/organisations'], "
            "a[href*='/organizations']"
        )
        if org_link:
            company = org_link.text(strip=True)

        # Full text content for extracting fields
        text = node.text(strip=True)

        # Country
        country = _extract_field(text, "country") or _extract_country_from_node(node)

        # Location
        location = _extract_location(node, text)

        # Description snippet
        description = _extract_description(node)

        # Research field
        research_field = _extract_field_link(node, "job_research_field")

        # Researcher profile → seniority
        profile = _extract_field_link(node, "job_research_profile")
        seniority = _map_researcher_profile(profile)

        # Posted date
        posted_at = _extract_date(text, "Posted on:")

        # Detect language from content
        language = "en"  # EURAXESS is primarily English

        return RawPosting(
            title=title,
            source="euraxess",
            company=company,
            url=url,
            description=description,
            location=location,
            country=country,
            language=language,
            seniority=seniority,
            posted_at=posted_at,
            source_id=source_id,
            raw_data={
                "research_field": research_field,
                "researcher_profile": profile,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_headers() -> dict[str, str]:
    return {
        "Accept": "text/html,application/xhtml+xml",
        "User-Agent": "Mozilla/5.0 (compatible; JobScout/1.0; research tool)",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _build_search_params(config: dict) -> dict[str, str | None]:
    """Build URL query params from source config."""
    params: dict[str, str | None] = {}

    # EURAXESS uses facet filters like f[0]=offer_type:job_offer
    filters = ["offer_type:job_offer"]  # Always filter to job offers

    countries = config.get("countries", [])
    for country in countries:
        filters.append(f"job_country:{country}")

    research_fields = config.get("research_fields", [])
    for field in research_fields:
        filters.append(f"job_research_field:{field}")

    for i, f in enumerate(filters):
        params[f"f[{i}]"] = f

    # Keyword search
    keyword = config.get("keyword")
    if keyword:
        params["search_api_fulltext"] = keyword

    return params


def _find_result_blocks(tree: HTMLParser) -> list:
    """Fallback: find result blocks by looking for heading links to /jobs/."""
    blocks = []
    for link in tree.css("a[href^='/jobs/']"):
        # Walk up to find the containing block
        parent = link.parent
        for _ in range(5):
            if parent and parent.tag in ("li", "div", "article"):
                if parent not in blocks:
                    blocks.append(parent)
                break
            if parent:
                parent = parent.parent
    return blocks


def _extract_field(text: str, keyword: str) -> str | None:
    """Extract a field value following a keyword label in text."""
    pattern = re.compile(rf"{keyword}\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _extract_country_from_node(node: Any) -> str | None:
    """Try to extract country from list item badges or text."""
    # EURAXESS shows country in a badge-like element near the top
    for el in node.css("span, div"):
        text = el.text(strip=True)
        if len(text) <= 30 and text in _EURAXESS_COUNTRIES:
            return _EURAXESS_COUNTRIES[text]
    return None


def _extract_location(node: Any, text: str) -> str | None:
    """Extract work location from the result node."""
    # Look for "Work Locations:" section
    loc_match = re.search(
        r"Work Locations?:.*?(?:Number of offers:\s*\d+,\s*)?(.+?)(?:Research Field|Researcher Profile|Funding|Application Deadline|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if loc_match:
        loc = loc_match.group(1).strip()
        # Clean up: remove extra whitespace
        loc = re.sub(r"\s+", " ", loc)
        # Truncate if too long (sometimes captures too much)
        if len(loc) > 200:
            loc = loc[:200].rsplit(",", 1)[0]
        return loc
    return None


def _extract_description(node: Any) -> str | None:
    """Extract the job description snippet."""
    # The snippet is typically in the main text block after the title
    for el in node.css("div.field--type-text-with-summary, div.field--name-body, p"):
        text = el.text(strip=True)
        if len(text) > 50:
            return text[:2000]
    # Fallback: grab all text, skip the first line (title) and metadata
    full = node.text(strip=True)
    if len(full) > 100:
        return full[:2000]
    return None


def _extract_field_link(node: Any, facet_name: str) -> str | None:
    """Extract a facet value from a link whose href contains the facet name."""
    link = node.css_first(f"a[href*='{facet_name}']")
    if link:
        return link.text(strip=True)
    return None


def _extract_date(text: str, label: str) -> str | None:
    """Extract a date string after a label like 'Posted on:'."""
    pattern = re.compile(rf"{label}\s*(\d{{1,2}}\s+\w+\s+\d{{4}})", re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _map_researcher_profile(profile: str | None) -> str | None:
    """Map EURAXESS R1-R4 researcher profiles to seniority."""
    if not profile:
        return None
    mapping = {
        "First Stage Researcher (R1)": "junior",
        "Recognised Researcher (R2)": "mid",
        "Established Researcher (R3)": "senior",
        "Leading Researcher (R4)": "lead",
        "Other Profession": None,
    }
    for key, value in mapping.items():
        if key in profile:
            return value
    return None


# Common EURAXESS countries → ISO codes
_EURAXESS_COUNTRIES: dict[str, str] = {
    "Austria": "AT", "Belgium": "BE", "Croatia": "HR", "Cyprus": "CY",
    "Czech Republic": "CZ", "Denmark": "DK", "Estonia": "EE",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR",
    "Hungary": "HU", "Iceland": "IS", "Ireland": "IE", "Israel": "IL",
    "Italy": "IT", "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU",
    "Malta": "MT", "Netherlands": "NL", "Norway": "NO", "Poland": "PL",
    "Portugal": "PT", "Romania": "RO", "Serbia": "RS", "Slovakia": "SK",
    "Slovenia": "SI", "Spain": "ES", "Sweden": "SE", "Switzerland": "CH",
    "Tunisia": "TN", "Türkiye": "TR", "United Kingdom": "GB",
}