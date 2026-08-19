"""Welcome to the Jungle adapter — via public Algolia API."""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
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

# --- Detail endpoint (M7d) -------------------------------------------------
#
# The search index carries no job description for any posting. What it carries
# under `profile` is a short requirements blurb, ~1KB where the real
# description runs 2-8KB, and 15% of hits leave even that null. The description
# lives here, one request per posting.
DETAIL_URL = (
    "https://api.welcometothejungle.com/api/v1/organizations/{org}/jobs/{job}"
)
DETAIL_HEADERS = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0",
}

# The detail endpoint splits a posting across three fields, and the requirements
# live in the second one. Measured over 25 payloads: `description` present 25/25
# (median 2699 chars), `profile` 22/25 (median 2029), `recruitment_process` 7/25
# (median 382). Taking `description` alone drops the section that carries the
# skills and the years requirement — a first cut of this did exactly that and
# halved skill extraction. Order follows how the page reads.
DETAIL_TEXT_FIELDS = ("description", "profile", "recruitment_process")

DEFAULT_FETCH_DETAILS = True
DEFAULT_DETAIL_CONCURRENCY = 8
DETAIL_TIMEOUT = 15.0        # per job: a slow posting must not stall the run
DETAIL_MAX_ATTEMPTS = 3      # on 429/5xx only; a 404 is an answer, not a failure
DETAIL_BACKOFF_BASE = 0.5    # seconds, doubled per attempt, with jitter
DETAIL_BACKOFF_JITTER = 0.25
# Above this share of failures the endpoint is throttling us rather than
# missing a few postings, and the run should say so loudly.
DETAIL_FAILURE_WARN_RATIO = 0.10

# Organisation slugs carry a legitimate numeric suffix ("wise-1", "front-1")
# and must be used as written: measured 10/10 succeed as-is, 0/10 stripped.
# The stripped form is only ever a retry, and only rescues companies that
# changed identity (earthcube-1 -> earthcube).
_ORG_SUFFIX_RE = re.compile(r"-\d+$")


HEADERS = {
    "x-algolia-application-id": ALGOLIA_APP_ID,
    "x-algolia-api-key": ALGOLIA_API_KEY,
    "content-type": "application/x-www-form-urlencoded",  # Algolia CORS quirk
    "accept": "*/*",
    "origin": WTTJ_SITE,
    "referer": WTTJ_SITE + "/",
}


@dataclass
class DetailReport:
    """What a run of the detail fetch achieved, so throttling is visible."""

    attempted: int = 0
    detail: int = 0           # real description, first slug
    renamed: int = 0          # real description, de-suffixed slug
    delisted: int = 0         # 404 on both slugs
    failed: int = 0           # everything else: timeouts, 429s, 5xx, bad bodies
    reasons: dict[str, int] | None = None

    def record(self, outcome: str) -> None:
        if self.reasons is None:
            self.reasons = {}
        self.reasons[outcome] = self.reasons.get(outcome, 0) + 1
        if outcome == "detail":
            self.detail += 1
        elif outcome == "detail-renamed":
            self.renamed += 1
        elif outcome == "delisted":
            self.delisted += 1
        else:
            self.failed += 1

    @property
    def resolved(self) -> int:
        """Postings that ended up with a real description."""
        return self.detail + self.renamed

    @property
    def failure_ratio(self) -> float:
        """Share that fell back for a reason other than being delisted.

        Delisted postings are excluded on purpose: a gone job is a correct
        answer, and counting it as failure would hide real throttling behind
        ordinary corpus decay.
        """
        return self.failed / self.attempted if self.attempted else 0.0

    def log(self) -> None:
        if not self.attempted:
            return
        logger.info(
            "wttj details: %d/%d resolved (%d renamed), %d delisted, %d failed (%.1f%%)",
            self.resolved, self.attempted, self.renamed,
            self.delisted, self.failed, 100.0 * self.failure_ratio,
        )
        if self.failure_ratio > DETAIL_FAILURE_WARN_RATIO:
            logger.warning(
                "wttj: %.1f%% of detail fetches failed (%d of %d), above the %.0f%% "
                "threshold — the endpoint is probably throttling. Reasons: %s. "
                "Set sources.wttj.fetch_details=false to fall back to index-only.",
                100.0 * self.failure_ratio, self.failed, self.attempted,
                100.0 * DETAIL_FAILURE_WARN_RATIO,
                ", ".join(
                    f"{k}={v}" for k, v in sorted(
                        (self.reasons or {}).items(), key=lambda kv: -kv[1],
                    )
                    if k not in ("detail", "detail-renamed", "delisted")
                ) or "none recorded",
            )


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

        if source_config.get("fetch_details", DEFAULT_FETCH_DETAILS):
            self.enrich_with_details(
                all_postings,
                concurrency=source_config.get(
                    "detail_concurrency", DEFAULT_DETAIL_CONCURRENCY,
                ),
            )
        else:
            logger.info(
                "wttj: fetch_details is off — storing the index blurb only. "
                "Descriptions will be ~1KB requirements text, not the posting.",
            )
        return all_postings

    # -- detail fetch -------------------------------------------------------

    def enrich_with_details(
        self,
        postings: list[RawPosting],
        concurrency: int = DEFAULT_DETAIL_CONCURRENCY,
        client: httpx.Client | None = None,
    ) -> "DetailReport":
        """Replace each posting's blurb with its real description, in parallel.

        Mutates the postings in place and returns what happened. Never raises:
        a posting whose detail fetch fails keeps the blurb it already had, so
        the worst case is the behaviour we had before this step existed.
        """
        report = DetailReport(attempted=len(postings))
        if not postings:
            return report

        concurrency = max(1, int(concurrency or DEFAULT_DETAIL_CONCURRENCY))
        owned = client is None
        client = client or httpx.Client(timeout=DETAIL_TIMEOUT, headers=DETAIL_HEADERS)
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for outcome in pool.map(
                    lambda p: self._apply_detail(client, p), postings,
                ):
                    report.record(outcome)
        finally:
            if owned:
                client.close()

        report.log()
        return report

    def _apply_detail(self, client: httpx.Client, posting: RawPosting) -> str:
        """Fetch one posting's description and write it onto the posting."""
        hit = posting.raw_data or {}
        org_slug = (hit.get("organization") or {}).get("slug")
        job_slug = hit.get("slug")
        if not org_slug or not job_slug:
            logger.info(
                "wttj: %s has no %s slug, cannot reach its detail page",
                posting.source_id, "organization" if not org_slug else "job",
            )
            return "no-slug"

        try:
            description, outcome = self._fetch_detail(client, org_slug, job_slug)
        except Exception as exc:
            # One posting must never cost the batch. httpx errors are already
            # handled below; this is the guard for everything unforeseen, so a
            # single malformed payload cannot lose 1300 other descriptions.
            logger.warning(
                "wttj detail %s/%s: unexpected %s: %s — falling back to the blurb",
                org_slug, job_slug, type(exc).__name__, exc,
            )
            return f"error-{type(exc).__name__}"

        if description:
            posting.description = description
            posting.description_source = "detail"
            return outcome

        if outcome == "delisted":
            # Both slugs 404: the posting is gone. Keep the row and the blurb —
            # what changes is that we now know the link is dead.
            posting.delisted_at = datetime.now(timezone.utc).isoformat()

        # Falling back. description_source stays whatever _hit_to_posting set,
        # which is 'blurb' when the index gave us text and 'none' when it did not.
        return outcome

    def _fetch_detail(
        self, client: httpx.Client, org_slug: str, job_slug: str,
    ) -> tuple[str | None, str]:
        """Try the slug as written, then the de-suffixed form. Returns (text, outcome)."""
        text, status = self._request_detail(client, org_slug, job_slug)
        if text:
            return text, "detail"
        if status != 404:
            return None, f"http-{status}"

        stripped = _ORG_SUFFIX_RE.sub("", org_slug)
        if stripped == org_slug:
            return None, "delisted"

        text, status = self._request_detail(client, stripped, job_slug)
        if text:
            logger.info(
                "wttj: %r 404s but %r resolves — the company was renamed",
                org_slug, stripped,
            )
            return text, "detail-renamed"
        if status == 404:
            return None, "delisted"
        return None, f"http-{status}"

    @staticmethod
    def _request_detail(
        client: httpx.Client, org_slug: str, job_slug: str,
    ) -> tuple[str | None, int | str]:
        """One detail URL, retried on 429/5xx. Returns (description, last status)."""
        url = DETAIL_URL.format(
            org=urllib.parse.quote(org_slug, safe=""),
            job=urllib.parse.quote(job_slug, safe=""),
        )
        status: int | str = "unreached"

        for attempt in range(DETAIL_MAX_ATTEMPTS):
            try:
                resp = client.get(url, timeout=DETAIL_TIMEOUT)
                status = resp.status_code
            except httpx.HTTPError as exc:
                status = type(exc).__name__
                logger.debug("wttj detail %s: %s", url, exc)
            else:
                if status == 404:
                    return None, 404
                if status == 200:
                    try:
                        job = (resp.json() or {}).get("job") or {}
                    except ValueError:
                        logger.warning("wttj detail %s: 200 but body is not JSON", url)
                        return None, "bad-json"
                    description = _join_detail_text(job)
                    if description:
                        return description, 200
                    # 200 with nothing in it is a schema surprise, not a miss.
                    logger.info(
                        "wttj detail %s: 200 but none of %s carried text",
                        url, ", ".join(DETAIL_TEXT_FIELDS),
                    )
                    return None, "empty-body"
                if status not in (429,) and not (500 <= int(status) < 600):
                    return None, status      # 4xx other than 404: not retryable

            if attempt + 1 < DETAIL_MAX_ATTEMPTS:
                delay = DETAIL_BACKOFF_BASE * (2 ** attempt)
                delay += random.uniform(0, DETAIL_BACKOFF_JITTER)
                logger.debug(
                    "wttj detail %s: status %s, retrying in %.2fs", url, status, delay,
                )
                time.sleep(delay)

        return None, status

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

        # The index's only text field. Flat string or null — never the
        # per-language dict the adapter used to defend against, and the index
        # carries no `recruitment_process` key at all. Both of those branches
        # were removed in M7d; they had never once fired. (The detail endpoint
        # does have recruitment_process — see DETAIL_TEXT_FIELDS.)
        #
        # This is a requirements blurb, not the description. The real text
        # comes from the detail endpoint; this is the fallback when that fails.
        blurb = hit.get("profile")
        if blurb is not None and not isinstance(blurb, str):
            # A shape change is a schema break, and schema breaks are how this
            # adapter has been wrong three times. Say so rather than coercing.
            logger.warning(
                "wttj: 'profile' on %s is %s, expected str or None — "
                "the index schema may have changed",
                hit.get("reference") or hit.get("objectID"), type(blurb).__name__,
            )
            blurb = None
        blurb = blurb.strip() if blurb else ""

        if not blurb:
            logger.info(
                "wttj: no blurb text for %s (%r) — 'profile' is %s; the detail "
                "fetch is this posting's only source of text",
                hit.get("reference") or hit.get("objectID"), title,
                "empty" if hit.get("profile") == "" else "null",
            )

        required_years = _guarded(
            hit, "has_experience_level_minimum", "experience_level_minimum", float,
        )
        salary_min = _guarded(
            hit, "has_salary_yearly_minimum", "salary_yearly_minimum", int,
        )

        return RawPosting(
            title=title,
            source="wttj",
            company=org.get("name"),
            url=url,
            description=blurb or None,
            # Provisional: the detail fetch overwrites both of these when it
            # runs. 'blurb' here means the index gave us something; the fetch
            # can only improve on it.
            description_source="blurb" if blurb else "none",
            location=office.get("city"),
            country=office.get("country_code") or office.get("country"),
            language=hit.get("language") or "en",
            posted_at=hit.get("published_at"),
            source_id=str(hit.get("objectID", "")),
            required_years_min=required_years,
            seniority_source="api" if required_years is not None else None,
            education_level=_guarded(hit, "has_education_level", "education_level", str),
            salary_yearly_min=salary_min,
            # A currency with no amount behind it is noise, not data.
            salary_currency=hit.get("salary_currency") if salary_min is not None else None,
            raw_data=hit,
        )


def _join_detail_text(job: dict[str, Any]) -> str:
    """Assemble the full posting from the detail payload's separate sections.

    Missing sections are skipped rather than defaulted: a posting with no
    recruitment process is normal, and an empty string would only add
    whitespace. A field of an unexpected type is a schema break and is logged.
    """
    parts: list[str] = []
    for field in DETAIL_TEXT_FIELDS:
        value = job.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            logger.warning(
                "wttj detail: %r is %s, expected str — the payload schema "
                "may have changed", field, type(value).__name__,
            )
            continue
        if value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def _guarded(hit: dict[str, Any], flag: str, field: str, cast: Any) -> Any:
    """Read an optional WTTJ field only when its has_* companion says it is set.

    WTTJ leaves the value key present but meaningless when the flag is false, so
    the flag is the only reliable signal. A genuine 0 (entry level, no salary
    floor) must survive, hence the explicit None check rather than truthiness.
    """
    if not hit.get(flag):
        return None
    value = hit.get(field)
    if value is None or value == "":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        logger.debug("wttj: unusable %s value %r", field, value)
        return None