"""M7d: real job descriptions from the WTTJ detail endpoint.

The Algolia index carries no description for any posting — only `profile`, a
~1KB requirements blurb that is null on 15% of hits. The real text needs one
request per posting, so these cover the slug rule, the fallback that must
never lose a blurb we already had, delisted detection, and the safety valves
around 1365 requests per run.
"""

from __future__ import annotations

import json

import httpx
import pytest

from jobscout.db import (
    init_db, migrate_m2, migrate_m3, migrate_m4, migrate_m5, migrate_m6,
    migrate_m7b, migrate_m7d,
)
from jobscout.sources import RawPosting, _insert_job, normalize
from jobscout.sources.wttj import (
    DETAIL_FAILURE_WARN_RATIO,
    DETAIL_MAX_ATTEMPTS,
    DetailReport,
    WTTJAdapter,
)

# Trailing whitespace stripped: the joiner strips each section, so a constant
# ending in a space would not compare equal to what comes back.
DESCRIPTION = ("About us. We are hiring a Data Engineer. " * 40).strip()
REQUIREMENTS = "Profil recherché : 3 ans d'expérience en Python."
PROCESS = "Two interviews and a take-home."
BLURB = "Vous disposez d'au moins 3 ans d'expérience."


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    """Backoff is real seconds. Tests assert on the delays, never wait them."""
    monkeypatch.setattr("jobscout.sources.wttj.time.sleep", lambda _: None)


@pytest.fixture
def db(tmp_path):
    conn = init_db(tmp_path / "t.db")
    for migrate in (migrate_m2, migrate_m3, migrate_m4, migrate_m5,
                    migrate_m6, migrate_m7b, migrate_m7d):
        migrate(conn)
    yield conn
    conn.close()


def hit(**overrides):
    """A WTTJ hit shaped like the real index response."""
    base = {
        "name": "Data Engineer",
        "objectID": "12345",
        "reference": "ACME_abc123",
        "slug": "data-engineer_paris",
        "organization": {"name": "Acme", "slug": "acme-1"},
        "office": {"city": "Paris", "country_code": "FR"},
        "language": "fr",
        "profile": BLURB,
    }
    base.update(overrides)
    return base


def posting(**overrides):
    """A RawPosting as _hit_to_posting would produce it."""
    return WTTJAdapter._hit_to_posting(hit(**overrides))


class Router:
    """An httpx transport that answers detail URLs from a routing table.

    Keys are (org_slug, job_slug); values are an int status, or (status, body).
    Records every org slug it was asked for, in order.
    """

    def __init__(self, routes, default=404):
        self.routes = routes
        self.default = default
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")
        org, job = parts[-3], parts[-1]
        self.calls.append(org)
        outcome = self.routes.get((org, job), self.default)
        status, body = outcome if isinstance(outcome, tuple) else (outcome, None)
        if status == 200 and body is None:
            body = {"job": {"description": DESCRIPTION}}
        return httpx.Response(status, json=body if body is not None else {})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


# ---------------------------------------------------------------------------
# Step 1 — the dead branches are gone
# ---------------------------------------------------------------------------

class TestDeadCodeRemoved:
    def test_a_flat_profile_string_is_the_blurb(self):
        assert posting().description == BLURB

    def test_a_dict_profile_is_refused_not_coerced(self, caplog):
        """The old per-language dict branch defended a schema WTTJ never served."""
        with caplog.at_level("WARNING"):
            p = posting(profile={"en": "We are looking for someone."})
        assert p.description is None
        assert p.description_source == "none"
        assert "schema may have changed" in caplog.text
        assert "ACME_abc123" in caplog.text

    def test_recruitment_process_is_ignored(self):
        """The key does not exist in this index; it must not resurface."""
        p = posting(recruitment_process="3 rounds of interviews.")
        assert "interviews" not in (p.description or "")

    def test_a_missing_blurb_is_logged_with_its_reference(self, caplog):
        with caplog.at_level("INFO"):
            p = posting(profile=None)
        assert p.description is None
        assert p.description_source == "none"
        assert "ACME_abc123" in caplog.text
        assert "null" in caplog.text

    def test_an_empty_blurb_says_empty_not_null(self, caplog):
        with caplog.at_level("INFO"):
            posting(profile="")
        assert "empty" in caplog.text


# ---------------------------------------------------------------------------
# Step 2 — the slug rule
# ---------------------------------------------------------------------------

class TestSlugRule:
    def test_the_suffixed_slug_is_used_as_written(self):
        router = Router({("acme-1", "data-engineer_paris"): 200})
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert router.calls == ["acme-1"]          # one request, no stripping
        assert p.description == DESCRIPTION
        assert p.description_source == "detail"

    def test_a_404_retries_once_with_the_suffix_stripped(self):
        router = Router({
            ("acme-1", "data-engineer_paris"): 404,
            ("acme", "data-engineer_paris"): 200,
        })
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert router.calls == ["acme-1", "acme"]
        assert p.description == DESCRIPTION
        assert p.description_source == "detail"
        assert p.delisted_at is None               # renamed, not gone

    def test_a_renamed_company_is_logged(self, caplog):
        router = Router({
            ("earthcube-1", "x_paris"): 404,
            ("earthcube", "x_paris"): 200,
        })
        p = posting(slug="x_paris", organization={"name": "Safran.AI", "slug": "earthcube-1"})
        with caplog.at_level("INFO"), router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert "renamed" in caplog.text

    def test_an_unsuffixed_slug_is_not_retried(self):
        router = Router({}, default=404)
        p = posting(organization={"name": "Acme", "slug": "acme"})
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert router.calls == ["acme"]             # nothing to strip

    def test_job_slugs_are_never_repaired(self):
        """Only the org slug has a retry form; the job slug is taken as given."""
        router = Router({}, default=404)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert {c for c in router.calls} == {"acme-1", "acme"}


# ---------------------------------------------------------------------------
# Delisted detection
# ---------------------------------------------------------------------------

class TestDelisted:
    def test_set_only_when_both_attempts_404(self):
        router = Router({}, default=404)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.delisted_at is not None
        assert p.delisted_at.endswith("+00:00")

    def test_not_set_when_the_stripped_retry_succeeds(self):
        router = Router({
            ("acme-1", "data-engineer_paris"): 404,
            ("acme", "data-engineer_paris"): 200,
        })
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.delisted_at is None

    def test_not_set_on_a_server_error(self):
        """A 500 means we could not tell. Absence of proof is not delisting."""
        router = Router({}, default=500)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.delisted_at is None
        assert p.description == BLURB

    def test_a_delisted_posting_keeps_its_blurb(self):
        router = Router({}, default=404)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description == BLURB
        assert p.description_source == "blurb"

    def test_it_survives_normalize_and_insert(self, db):
        router = Router({}, default=404)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        _insert_job(db, normalize(p))
        db.commit()
        row = db.execute("SELECT delisted_at, description_source FROM jobs").fetchone()
        assert row["delisted_at"] is not None
        assert row["description_source"] == "blurb"


# ---------------------------------------------------------------------------
# Fallback: never lose text we already had
# ---------------------------------------------------------------------------

class TestFallback:
    @pytest.mark.parametrize("status", [404, 429, 500, 503])
    def test_a_failed_fetch_falls_back_to_the_blurb(self, status):
        router = Router({}, default=status)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description == BLURB
        assert p.description_source == "blurb"

    def test_a_transport_error_falls_back_too(self):
        def boom(request):
            raise httpx.ConnectError("no route to host")

        p = posting()
        with httpx.Client(transport=httpx.MockTransport(boom)) as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description == BLURB

    def test_a_row_with_no_blurb_stays_none_not_empty(self):
        router = Router({}, default=404)
        p = posting(profile=None)
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description is None
        assert p.description_source == "none"

    def test_a_200_with_an_empty_body_is_not_a_description(self):
        router = Router({("acme-1", "data-engineer_paris"): (200, {"job": {"description": ""}})})
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description == BLURB
        assert p.description_source == "blurb"

    def test_one_bad_posting_does_not_cost_the_batch(self, monkeypatch):
        """A single unforeseen error must not lose the other 1300 descriptions."""
        adapter = WTTJAdapter()
        real = adapter._fetch_detail
        calls = {"n": 0}

        def flaky(client, org, job):
            calls["n"] += 1
            if job == "job-1":
                raise RuntimeError("something unforeseen")
            return real(client, org, job)

        monkeypatch.setattr(adapter, "_fetch_detail", flaky)
        router = Router({("acme-1", f"job-{i}"): 200 for i in range(3)})
        postings = [posting(slug=f"job-{i}", objectID=str(i)) for i in range(3)]
        with router.client() as client:
            report = adapter.enrich_with_details(postings, client=client)

        assert calls["n"] == 3
        assert report.detail == 2                      # the other two survived
        assert report.failed == 1
        assert postings[1].description == BLURB        # the bad one fell back

    def test_a_posting_without_slugs_is_skipped_cleanly(self):
        p = posting(slug=None)
        router = Router({})
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert router.calls == []
        assert p.description == BLURB


# ---------------------------------------------------------------------------
# description_source, all three paths
# ---------------------------------------------------------------------------

class TestDetailTextAssembly:
    """The detail payload splits a posting; the requirements are in `profile`."""

    def _fetch(self, body):
        router = Router({("acme-1", "data-engineer_paris"): (200, body)})
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        return p

    def test_all_three_sections_are_joined(self):
        p = self._fetch({"job": {
            "description": DESCRIPTION,
            "profile": REQUIREMENTS,
            "recruitment_process": PROCESS,
        }})
        assert p.description == "\n\n".join([DESCRIPTION, REQUIREMENTS, PROCESS])
        assert p.description_source == "detail"

    def test_the_requirements_section_is_never_dropped(self):
        """Taking `description` alone halved skill extraction on the corpus."""
        p = self._fetch({"job": {"description": DESCRIPTION, "profile": REQUIREMENTS}})
        assert REQUIREMENTS in p.description

    def test_missing_sections_are_skipped_not_padded(self):
        p = self._fetch({"job": {
            "description": DESCRIPTION, "profile": None, "recruitment_process": "",
        }})
        assert p.description == DESCRIPTION

    def test_requirements_alone_still_counts_as_a_detail(self):
        p = self._fetch({"job": {"description": None, "profile": REQUIREMENTS}})
        assert p.description == REQUIREMENTS
        assert p.description_source == "detail"

    def test_a_non_string_section_is_logged_and_skipped(self, caplog):
        with caplog.at_level("WARNING"):
            p = self._fetch({"job": {
                "description": DESCRIPTION, "profile": {"en": "wrong shape"},
            }})
        assert p.description == DESCRIPTION
        assert "payload schema" in caplog.text

    def test_every_section_empty_falls_back_to_the_blurb(self):
        p = self._fetch({"job": {"description": "", "profile": None}})
        assert p.description == BLURB
        assert p.description_source == "blurb"


class TestDescriptionSource:
    def test_detail(self):
        router = Router({("acme-1", "data-engineer_paris"): 200})
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description_source == "detail"

    def test_blurb(self):
        router = Router({}, default=500)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description_source == "blurb"

    def test_none(self):
        router = Router({}, default=500)
        p = posting(profile=None)
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        assert p.description_source == "none"

    def test_all_three_round_trip_through_the_db(self, db):
        for i, (src, desc) in enumerate([
            ("detail", "a real description"), ("blurb", BLURB), ("none", None),
        ]):
            _insert_job(db, normalize(RawPosting(
                title=f"t{i}", source="wttj", url=f"https://x/{i}", source_id=str(i),
                description=desc, description_source=src,
            )))
        db.commit()
        stored = dict(db.execute(
            "SELECT description_source, COUNT(*) FROM jobs GROUP BY 1"
        ).fetchall())
        assert stored == {"detail": 1, "blurb": 1, "none": 1}


# ---------------------------------------------------------------------------
# Safety valves
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def _fetch(self, monkeypatch, config):
        adapter = WTTJAdapter()
        monkeypatch.setattr(
            adapter, "_search_keyword",
            lambda *a, **k: [posting()],
        )
        called = []
        monkeypatch.setattr(
            adapter, "enrich_with_details",
            lambda *a, **k: called.append(True) or DetailReport(),
        )
        adapter.fetch(config)
        return called

    def test_details_are_fetched_by_default(self, monkeypatch):
        assert self._fetch(monkeypatch, {"keywords": ["data"]}) == [True]

    def test_the_switch_turns_them_off(self, monkeypatch):
        assert self._fetch(
            monkeypatch, {"keywords": ["data"], "fetch_details": False},
        ) == []

    def test_the_switch_off_still_returns_postings(self, monkeypatch):
        adapter = WTTJAdapter()
        monkeypatch.setattr(adapter, "_search_keyword", lambda *a, **k: [posting()])
        out = adapter.fetch({"keywords": ["data"], "fetch_details": False})
        assert len(out) == 1
        assert out[0].description == BLURB
        assert out[0].description_source == "blurb"

    def test_the_shipped_config_leaves_it_on(self):
        from jobscout.config import DEFAULTS

        wttj = next(s for s in DEFAULTS["sources"] if s["adapter"] == "wttj")
        assert wttj["fetch_details"] is True
        assert wttj["detail_concurrency"] == 8


class TestBackoff:
    def _count_attempts(self, status, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr("jobscout.sources.wttj.time.sleep", slept.append)
        router = Router({}, default=status)
        p = posting()
        with router.client() as client:
            WTTJAdapter().enrich_with_details([p], client=client)
        return len(router.calls), slept

    def test_429_is_retried_to_the_attempt_cap(self, monkeypatch):
        attempts, slept = self._count_attempts(429, monkeypatch)
        # One org slug, tried DETAIL_MAX_ATTEMPTS times; no stripped retry,
        # because a 429 is not a 404.
        assert attempts == DETAIL_MAX_ATTEMPTS
        assert len(slept) == DETAIL_MAX_ATTEMPTS - 1

    def test_the_delay_grows(self, monkeypatch):
        _, slept = self._count_attempts(429, monkeypatch)
        assert slept[1] > slept[0]

    def test_5xx_is_retried(self, monkeypatch):
        attempts, _ = self._count_attempts(503, monkeypatch)
        assert attempts == DETAIL_MAX_ATTEMPTS

    def test_404_is_not_retried(self, monkeypatch):
        """A 404 is an answer. Retrying it would triple the cost of delisting."""
        slept: list[float] = []
        monkeypatch.setattr("jobscout.sources.wttj.time.sleep", slept.append)
        router = Router({}, default=404)
        with router.client() as client:
            WTTJAdapter().enrich_with_details([posting()], client=client)
        assert router.calls == ["acme-1", "acme"]   # one each, not three
        assert slept == []

    def test_a_non_retryable_4xx_stops_immediately(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr("jobscout.sources.wttj.time.sleep", slept.append)
        router = Router({}, default=403)
        with router.client() as client:
            WTTJAdapter().enrich_with_details([posting()], client=client)
        assert router.calls == ["acme-1"]
        assert slept == []


class TestFailureReporting:
    def test_the_report_counts_each_outcome(self):
        report = DetailReport(attempted=4)
        for outcome in ("detail", "detail-renamed", "delisted", "http-500"):
            report.record(outcome)
        assert (report.detail, report.renamed, report.delisted, report.failed) == (1, 1, 1, 1)
        assert report.resolved == 2

    def test_delisted_is_not_counted_as_failure(self):
        """Corpus decay must not masquerade as throttling."""
        report = DetailReport(attempted=10)
        for _ in range(9):
            report.record("delisted")
        report.record("detail")
        assert report.failure_ratio == 0.0

    def test_a_high_failure_rate_warns_prominently(self, caplog):
        report = DetailReport(attempted=10)
        for _ in range(3):
            report.record("http-429")
        for _ in range(7):
            report.record("detail")
        assert report.failure_ratio > DETAIL_FAILURE_WARN_RATIO
        with caplog.at_level("WARNING"):
            report.log()
        assert "throttling" in caplog.text
        assert "fetch_details=false" in caplog.text

    def test_an_ordinary_rate_does_not_warn(self, caplog):
        report = DetailReport(attempted=100)
        for _ in range(5):
            report.record("http-500")
        for _ in range(95):
            report.record("detail")
        with caplog.at_level("WARNING"):
            report.log()
        assert "throttling" not in caplog.text

    def test_the_run_reports_what_it_achieved(self):
        router = Router({
            ("acme-1", "a"): 200,
            ("acme-1", "b"): 404, ("acme", "b"): 200,
            ("acme-1", "c"): 404, ("acme", "c"): 404,
        })
        postings = [posting(slug=s, objectID=s) for s in ("a", "b", "c")]
        with router.client() as client:
            report = WTTJAdapter().enrich_with_details(postings, client=client)
        assert (report.attempted, report.detail, report.renamed,
                report.delisted, report.failed) == (3, 1, 1, 1, 0)


class TestConcurrency:
    def test_every_posting_is_fetched_once(self):
        router = Router({("acme-1", f"job-{i}"): 200 for i in range(25)})
        postings = [posting(slug=f"job-{i}", objectID=str(i)) for i in range(25)]
        with router.client() as client:
            report = WTTJAdapter().enrich_with_details(
                postings, concurrency=8, client=client,
            )
        assert len(router.calls) == 25
        assert report.detail == 25
        assert all(p.description == DESCRIPTION for p in postings)

    def test_an_empty_batch_is_a_no_op(self):
        report = WTTJAdapter().enrich_with_details([])
        assert report.attempted == 0

    def test_a_zero_concurrency_setting_does_not_hang(self):
        router = Router({("acme-1", "data-engineer_paris"): 200})
        with router.client() as client:
            WTTJAdapter().enrich_with_details([posting()], concurrency=0, client=client)
        assert router.calls == ["acme-1"]


class TestMigration:
    def test_the_columns_exist(self, db):
        cols = {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
        assert {"description_source", "delisted_at"} <= cols

    def test_it_is_idempotent(self, db):
        migrate_m7d(db)
        migrate_m7d(db)          # must not raise

    def test_adapters_that_say_nothing_store_null(self, db):
        _insert_job(db, normalize(RawPosting(
            title="t", source="eures", url="https://x/1", source_id="1",
        )))
        db.commit()
        row = db.execute("SELECT description_source, delisted_at FROM jobs").fetchone()
        assert row["description_source"] is None
        assert row["delisted_at"] is None


# ---------------------------------------------------------------------------
# Cleanup pass: migration splitting, position type, EURES language filter
# ---------------------------------------------------------------------------

class TestMigrationStatementSplitting:
    """A semicolon inside a comment is not a statement separator."""

    def test_a_semicolon_in_a_comment_does_not_split_the_statement(self):
        from jobscout.db import split_statements

        sql = """
        -- keeps its row and its score; what changes is only the link
        ALTER TABLE jobs ADD COLUMN widget TEXT;
        """
        assert split_statements(sql) == ["ALTER TABLE jobs ADD COLUMN widget TEXT"]

    def test_such_a_migration_actually_runs(self, db):
        from jobscout.db import _add_missing_columns

        _add_missing_columns(db, """
        -- one; two; three semicolons in this comment
        ALTER TABLE jobs ADD COLUMN semicolon_canary TEXT;
        -- and another; here
        ALTER TABLE jobs ADD COLUMN second_canary TEXT;
        """)
        cols = {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
        assert {"semicolon_canary", "second_canary"} <= cols

    def test_trailing_comment_after_the_last_statement(self):
        from jobscout.db import split_statements

        assert split_statements(
            "ALTER TABLE jobs ADD COLUMN a TEXT;  -- done; finished"
        ) == ["ALTER TABLE jobs ADD COLUMN a TEXT"]

    def test_comments_are_stripped_from_within_a_statement(self):
        from jobscout.db import split_statements

        statements = split_statements(
            "ALTER TABLE jobs   -- why; because\n    ADD COLUMN a TEXT;"
        )
        assert len(statements) == 1
        assert "because" not in statements[0]
        assert statements[0].split() == [
            "ALTER", "TABLE", "jobs", "ADD", "COLUMN", "a", "TEXT",
        ]

    def test_the_shipped_migrations_all_parse(self):
        from jobscout.db import (
            MIGRATE_M2_SQL, MIGRATE_M3_SQL, MIGRATE_M7B_SQL, MIGRATE_M7D_SQL,
            split_statements,
        )

        for sql in (MIGRATE_M2_SQL, MIGRATE_M3_SQL, MIGRATE_M7B_SQL, MIGRATE_M7D_SQL):
            for statement in split_statements(sql):
                assert statement.upper().startswith("ALTER TABLE"), statement
                assert "--" not in statement


class TestPositionTypeWordBoundaries:
    @pytest.mark.parametrize("title,description", [
        ("Lead Analytics Engineer", "We build internal and international tooling."),
        ("Senior Data Engineer", "Notre mission est de transformer la donnée."),
        ("Backend Developer", "You will join our internal platform team."),
        ("Data Scientist", "An international, mission-driven company."),
    ])
    def test_substrings_no_longer_misclassify(self, title, description):
        from jobscout.sources import classify_position_type

        assert classify_position_type(title, description) == "job"

    @pytest.mark.parametrize("title,expected", [
        ("Stage - Data Scientist", "internship"),
        ("Stagiaire Data Analyst", "internship"),
        ("Werkstudent Data (m/w/d)", "internship"),
        ("Praktikum Machine Learning", "internship"),
        ("Alternance Data Analyst", "internship"),
        ("Data Engineering Internship", "internship"),
        ("Freelance Backend Developer", "freelance"),
        ("Développeur Freelance", "freelance"),
        ("Senior Data Engineer", "job"),
    ])
    def test_real_markers_still_match(self, title, expected):
        from jobscout.sources import classify_position_type

        assert classify_position_type(title, "") == expected

    def test_a_title_marker_beats_a_conflicting_description_marker(self):
        """Title first. A description-only marker still classifies, as before:
        this pass changed how keywords match, not where they are looked for."""
        from jobscout.sources import classify_position_type

        assert classify_position_type(
            "Freelance Data Engineer", "Nous proposons aussi des stages.",
        ) == "freelance"
        assert classify_position_type(
            "Senior Data Engineer", "We also run an internship programme.",
        ) == "internship"

    def test_a_hyphenated_phrase_matches(self):
        from jobscout.sources import classify_position_type

        assert classify_position_type("Working-Student Data", "") == "internship"


class TestEURESLanguageFilter:
    def _posting(self, language):
        return RawPosting(
            title="Data Engineer", source="eures", url=f"https://x/{language}",
            source_id=language, language=language,
        )

    def _fetch(self, monkeypatch, config, languages):
        from jobscout.sources.eures import EURESAdapter

        adapter = EURESAdapter()
        monkeypatch.setattr(
            adapter, "_search_keyword",
            lambda *a, **k: [self._posting(lang) for lang in languages],
        )
        return [p.language for p in adapter.fetch(config)]

    def test_the_default_keeps_fr_en_de(self, monkeypatch):
        kept = self._fetch(
            monkeypatch, {"keywords": ["data"]}, ["fr", "en", "de", "nl", "sv", "sk"],
        )
        assert kept == ["fr", "en", "de"]

    def test_an_explicit_list_is_honoured(self, monkeypatch):
        kept = self._fetch(
            monkeypatch, {"keywords": ["data"], "languages": ["nl"]},
            ["fr", "nl", "de"],
        )
        assert kept == ["nl"]

    def test_null_keeps_everything(self, monkeypatch):
        kept = self._fetch(
            monkeypatch, {"keywords": ["data"], "languages": None},
            ["fr", "nl", "sk"],
        )
        assert kept == ["fr", "nl", "sk"]

    def test_an_empty_list_also_keeps_everything(self, monkeypatch):
        kept = self._fetch(
            monkeypatch, {"keywords": ["data"], "languages": []}, ["fr", "nl"],
        )
        assert kept == ["fr", "nl"]

    def test_region_tags_and_case_are_tolerated(self, monkeypatch):
        kept = self._fetch(
            monkeypatch, {"keywords": ["data"]}, ["FR", "de-DE", "nl-BE"],
        )
        assert kept == ["FR", "de-DE"]

    def test_the_shipped_config_defaults_to_fr_en_de(self):
        from jobscout.config import DEFAULTS

        eures = next(s for s in DEFAULTS["sources"] if s["adapter"] == "eures")
        assert eures["languages"] == ["fr", "en", "de"]


# ---------------------------------------------------------------------------
# Vocabulary matching: word order, plurals, case
# ---------------------------------------------------------------------------

class TestRestWordOrder:
    @pytest.mark.parametrize("text", [
        "Développer des API REST",
        "Déploiement de modèles en production (APIs REST, monitoring)",
        "Capacité à concevoir et développer des API RESTful",
        "Design and build RESTful APIs",
        "Deploy REST APIs at scale",
        "REST-Schnittstelle bauen",
        "Building Web APIs",
        "REST-API und Microservices",
    ])
    def test_every_surface_form_yields_the_canonical_skill(self, text):
        from jobscout.profiles import RuleBasedExtractor

        assert "rest api" in RuleBasedExtractor().extract_from_text(text)["skills"]

    def test_the_canonical_string_is_what_gets_reported(self):
        """French and English postings must produce the same facet text."""
        from jobscout.profiles import RuleBasedExtractor

        extractor = RuleBasedExtractor()
        assert (extractor.extract_from_text("des API REST")["skills"]
                == extractor.extract_from_text("REST APIs")["skills"]
                == ["rest api"])


class TestPluralRelaxation:
    @pytest.mark.parametrize("text,expected", [
        ("nos data lakes", "data lake"),
        ("two feature stores", "feature store"),
        ("modern data warehouses", "data warehouse"),
        ("cloud architectures", "cloud architecture"),
        ("intégration et déploiement continus", "ci/cd"),
    ])
    def test_a_trailing_plural_still_matches(self, text, expected):
        from jobscout.profiles import RuleBasedExtractor

        assert expected in RuleBasedExtractor().extract_from_text(text)["skills"]

    def test_the_singular_still_matches(self):
        from jobscout.profiles import RuleBasedExtractor

        assert "data lake" in RuleBasedExtractor().extract_from_text(
            "a single data lake",
        )["skills"]

    def test_it_does_not_swallow_a_longer_word(self):
        """The relaxation is one optional suffix, not a prefix match."""
        from jobscout.profiles import skill_pattern

        assert not skill_pattern("data lake").search("data lakeshore property")


class TestCaseHandling:
    def test_patterns_are_case_insensitive(self):
        from jobscout.profiles import SKILL_PATTERNS, skill_pattern

        assert SKILL_PATTERNS["rest api"].search("REST API")
        assert skill_pattern("kubernetes").search("Kubernetes")

    def test_matching_no_longer_depends_on_the_caller_lowering(self):
        from jobscout.profiles import RuleBasedExtractor

        skills = RuleBasedExtractor().extract_from_text(
            "PYTHON, Docker AND Kubernetes",
        )["skills"]
        assert {"python", "docker", "kubernetes"} <= set(skills)

    def test_a_non_lowered_first_argument_is_logged(self, caplog):
        """The two-argument contract is order-sensitive and easy to get wrong."""
        from jobscout.profiles import RuleBasedExtractor

        with caplog.at_level("WARNING"):
            RuleBasedExtractor()._extract_skills("Python AND Docker", "x")
        assert "not lowercased" in caplog.text
        assert "swapped" in caplog.text

    def test_correct_usage_logs_nothing(self, caplog):
        from jobscout.profiles import RuleBasedExtractor

        with caplog.at_level("WARNING"):
            RuleBasedExtractor().extract_from_text("Python and Docker")
        assert "not lowercased" not in caplog.text


class TestPruneCommand:
    """Dry run by default; --apply removes the rows and their matches."""

    def _setup(self, tmp_path, monkeypatch):
        import argparse

        from jobscout.db import upsert_profile

        config = {
            "db_path": str(tmp_path / "prune.db"),
            "sources": [
                {"adapter": "eures", "languages": ["fr", "en", "de"]},
                {"adapter": "wttj"},          # no filter declared: untouched
            ],
        }
        monkeypatch.setattr("jobscout.cli.load_config", lambda *_a, **_k: config)

        from jobscout.cli import _setup_db

        conn, _ = _setup_db(config)
        upsert_profile(conn, {
            "name": "default", "raw_text": "", "skills": [], "domains": [],
            "seniority": "junior", "languages": [], "target_locations": [],
            "company_types": [], "position_types": [],
        })
        profile_id = conn.execute("SELECT id FROM profiles").fetchone()[0]
        for i, (source, language) in enumerate([
            ("eures", "fr"), ("eures", "nl"), ("eures", "sv"), ("eures", None),
            ("wttj", "nl"),                   # other source, filter not declared
        ]):
            _insert_job(conn, normalize(RawPosting(
                title=f"t{i}", source=source, url=f"https://x/{i}",
                source_id=str(i), language=language,
            )))
        for job_id, in conn.execute("SELECT id FROM jobs").fetchall():
            conn.execute(
                "INSERT INTO matches (profile_id, job_id, score) VALUES (?, ?, 0.5)",
                (profile_id, job_id),
            )
        conn.commit()
        conn.close()
        return argparse.Namespace(config="config.yaml", apply=False)

    def test_the_dry_run_deletes_nothing(self, tmp_path, monkeypatch, capsys):
        import sqlite3

        from jobscout.cli import cmd_prune

        args = self._setup(tmp_path, monkeypatch)
        cmd_prune(args)
        out = capsys.readouterr().out

        conn = sqlite3.connect(tmp_path / "prune.db")
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 5
        conn.close()
        assert "DRY RUN" in out
        assert "Failing the filter: 2" in out

    def test_apply_removes_the_rows_and_their_matches(self, tmp_path, monkeypatch):
        import sqlite3

        from jobscout.cli import cmd_prune

        args = self._setup(tmp_path, monkeypatch)
        args.apply = True
        cmd_prune(args)

        conn = sqlite3.connect(tmp_path / "prune.db")
        conn.row_factory = sqlite3.Row
        left = [(r["source"], r["language"])
                for r in conn.execute("SELECT source, language FROM jobs")]
        assert sorted(left, key=lambda x: (x[0], x[1] or "")) == [
            ("eures", None), ("eures", "fr"), ("wttj", "nl"),
        ]
        assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 3
        conn.close()

    def test_a_source_without_a_declared_filter_is_untouched(
        self, tmp_path, monkeypatch,
    ):
        import sqlite3

        from jobscout.cli import cmd_prune

        args = self._setup(tmp_path, monkeypatch)
        args.apply = True
        cmd_prune(args)
        conn = sqlite3.connect(tmp_path / "prune.db")
        assert conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE source = 'wttj'"
        ).fetchone()[0] == 1
        conn.close()

    def test_an_undetected_language_is_kept_not_guessed(
        self, tmp_path, monkeypatch, capsys,
    ):
        """No detected language is not a failed filter."""
        from jobscout.cli import cmd_prune

        args = self._setup(tmp_path, monkeypatch)
        cmd_prune(args)
        assert "undetected, kept" in capsys.readouterr().out


class TestPruneProtectsTheGoldSet:
    """A row the evaluation set names must survive the prune.

    Deleting one does not fail loudly - it leaves a gold entry pointing at a
    missing id, and the eval silently scores one fewer posting. These tests
    fail if a gold-referenced id is ever selected for deletion.
    """

    def _setup(self, tmp_path, monkeypatch, gold_ids):
        import argparse
        import textwrap

        from jobscout.db import upsert_profile

        gold = tmp_path / "skills_gold.yaml"
        gold.write_text(
            "entries:\n" + "".join(
                textwrap.dedent(f"""\
                - job_id: {job_id}
                  title: t
                  expected_skills: []
                """) for job_id in gold_ids
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("jobscout.cli.GOLD_SET_PATH", gold)

        config = {
            "db_path": str(tmp_path / "prune.db"),
            "sources": [{"adapter": "eures", "languages": ["fr", "en", "de"]}],
        }
        monkeypatch.setattr("jobscout.cli.load_config", lambda *_a, **_k: config)

        from jobscout.cli import _setup_db

        conn, _ = _setup_db(config)
        upsert_profile(conn, {
            "name": "default", "raw_text": "", "skills": [], "domains": [],
            "seniority": "junior", "languages": [], "target_locations": [],
            "company_types": [], "position_types": [],
        })
        # Three rows fail the filter; the caller decides which are gold.
        for i, language in enumerate(["nl", "it", "sv", "fr"]):
            _insert_job(conn, normalize(RawPosting(
                title=f"t{i}", source="eures", url=f"https://x/{i}",
                source_id=str(i), language=language,
            )))
        conn.commit()
        ids = [r[0] for r in conn.execute("SELECT id FROM jobs ORDER BY id")]
        conn.close()
        return argparse.Namespace(config="config.yaml", apply=False), ids

    def test_a_gold_row_is_never_deleted(self, tmp_path, monkeypatch):
        import sqlite3

        from jobscout.cli import cmd_prune

        args, ids = self._setup(tmp_path, monkeypatch, gold_ids=[])
        # Re-point the gold set at the first failing row now that ids exist.
        gold = tmp_path / "skills_gold.yaml"
        gold.write_text(
            f"entries:\n- job_id: {ids[0]}\n  title: t\n  expected_skills: []\n",
            encoding="utf-8",
        )
        args.apply = True
        cmd_prune(args)

        conn = sqlite3.connect(tmp_path / "prune.db")
        live = {r[0] for r in conn.execute("SELECT id FROM jobs")}
        conn.close()
        assert ids[0] in live, "prune deleted a row the gold set references"
        assert ids[1] not in live and ids[2] not in live

    def test_the_skipped_ids_are_reported(self, tmp_path, monkeypatch, capsys):
        from jobscout.cli import cmd_prune

        args, ids = self._setup(tmp_path, monkeypatch, gold_ids=[])
        gold = tmp_path / "skills_gold.yaml"
        gold.write_text(
            f"entries:\n- job_id: {ids[0]}\n  title: t\n  expected_skills: []\n",
            encoding="utf-8",
        )
        cmd_prune(args)
        out = capsys.readouterr().out
        assert "Protected by the gold set, not deleted: 1" in out
        assert str(ids[0]) in out
        # The headline count must exclude them, not merely mention them.
        assert "Failing the filter: 2" in out

    def test_an_unparseable_gold_set_aborts_rather_than_deleting(
        self, tmp_path, monkeypatch,
    ):
        """It may name rows about to go, and we cannot tell which."""
        import sqlite3

        import pytest

        from jobscout.cli import cmd_prune

        args, ids = self._setup(tmp_path, monkeypatch, gold_ids=[])
        (tmp_path / "skills_gold.yaml").write_text(
            "entries:\n- job_id: 1\n   bad: indent\n- oops\n", encoding="utf-8",
        )
        args.apply = True
        with pytest.raises(SystemExit):
            cmd_prune(args)

        conn = sqlite3.connect(tmp_path / "prune.db")
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == len(ids)
        conn.close()

    def test_a_missing_gold_set_is_not_an_error(self, tmp_path, monkeypatch):
        from jobscout.cli import cmd_prune, gold_referenced_job_ids

        args, _ = self._setup(tmp_path, monkeypatch, gold_ids=[])
        monkeypatch.setattr(
            "jobscout.cli.GOLD_SET_PATH", tmp_path / "does_not_exist.yaml",
        )
        assert gold_referenced_job_ids(tmp_path / "does_not_exist.yaml") == set()
        cmd_prune(args)          # must not raise

    def test_a_placeholder_job_id_protects_nothing_and_warns(
        self, tmp_path, monkeypatch, caplog,
    ):
        from jobscout.cli import gold_referenced_job_ids

        gold = tmp_path / "gold.yaml"
        gold.write_text(
            "entries:\n- job_id: <id>\n  expected_skills: []\n"
            "- job_id: 7\n  expected_skills: []\n",
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            assert gold_referenced_job_ids(gold) == {7}
        assert "protects no row" in caplog.text

    def test_the_real_gold_set_resolves_to_live_rows(self):
        """Guards the invariant the other tests enforce mechanically."""
        from jobscout.cli import GOLD_SET_PATH, gold_referenced_job_ids

        if not GOLD_SET_PATH.exists():
            import pytest

            pytest.skip("no gold set checked out")
        ids = gold_referenced_job_ids()
        assert ids, "gold set names no job ids"
        assert all(isinstance(i, int) for i in ids)


class TestApiZeroIsNotAStatedRequirement:
    """WTTJ sends experience_level_minimum: 0 when a posting states nothing.

    The field therefore carries "unknown" and "genuinely zero" in one value.
    Read literally it terminates the fallback chain at `api`, so the posting's
    own prose is never consulted and the UI reports "stated by the source".
    """

    def _job(self, **over):
        job = {
            "required_years_min": 0.0,
            "description": "",
            "title": "Data Engineer",
            "language": "en",
        }
        job.update(over)
        return job

    def test_a_zero_falls_through_to_the_description(self):
        from jobscout.matching import resolve_seniority

        verdict = resolve_seniority(
            self._job(description="We ask for 5 years of experience in ML."),
            candidate_years=5.0,
        )
        assert verdict.required_years == 5.0
        assert verdict.source == "description"
        assert verdict.snippet is not None

    def test_a_silent_description_reports_none_not_zero(self):
        from jobscout.matching import resolve_seniority

        verdict = resolve_seniority(
            self._job(description="We build data pipelines.", title="Data Engineer"),
            candidate_years=5.0,
        )
        assert verdict.required_years is None
        assert verdict.source == "none"
        assert verdict.multiplier == 1.0
        assert verdict.filtered is False

    def test_a_zero_still_reaches_the_title_layer(self):
        from jobscout.matching import resolve_seniority

        verdict = resolve_seniority(
            self._job(title="Senior Data Engineer"), candidate_years=1.0,
        )
        assert verdict.source == "title"
        assert verdict.required_years is not None

    def test_the_fallthrough_is_logged(self, caplog):
        from jobscout.matching import resolve_seniority

        with caplog.at_level("INFO"):
            resolve_seniority(self._job(), candidate_years=5.0)
        assert "required_years_min is 0" in caplog.text
        assert "treating it as unset" in caplog.text

    def test_a_real_nonzero_api_value_is_untouched(self):
        from jobscout.matching import resolve_seniority

        verdict = resolve_seniority(
            self._job(required_years_min=3.0,
                      description="We ask for 5 years of experience."),
            candidate_years=5.0,
        )
        assert verdict.required_years == 3.0
        assert verdict.source == "api"

    def test_a_null_api_value_behaves_as_before(self):
        from jobscout.matching import resolve_seniority

        verdict = resolve_seniority(
            self._job(required_years_min=None,
                      description="We ask for 4 years of experience."),
            candidate_years=5.0,
        )
        assert verdict.required_years == 4.0
        assert verdict.source == "description"


class TestJob98Regression:
    """Bluecoders "ML Ops" — stored as api 0.0 while stating 5+ years twice.

    The text is the posting's own, verbatim. The 0-sentinel fix stops it being
    reported as a stated zero; it does not yet recover 5.0, because both
    statements put the word "experience" 78 characters from the number and the
    proximity guard allows 60. That second cause is deliberately not patched
    here — widening the window is a corpus-wide change, not a fix to this row.
    """

    TEXT = (
        "70-80 K€ gross per year + ~5% bonus + profit sharing. Paris (hybrid, "
        "3 days on-site / week). Remote: 2 days / week after onboarding. "
        "English: fluent / French: nice to have. 5+ years in ML Engineering / "
        "MLOps / Software Engineering with strong ML in production. "
        "Who we are looking for : 5+ years as ML Engineer / MLOps / Software "
        "Engineer with strong ML production experience. Proven track record "
        "putting ML models into production and running them reliably."
    )

    def _job(self):
        return {
            "required_years_min": 0.0, "description": self.TEXT,
            "title": "ML Ops", "language": "fr",
        }

    def test_it_is_no_longer_reported_as_a_stated_zero(self):
        from jobscout.matching import resolve_seniority

        verdict = resolve_seniority(self._job(), candidate_years=5.0)
        assert verdict.source != "api", (
            "an API 0 must not be presented as a stated requirement")
        assert verdict.required_years != 0.0

    def test_it_is_not_filtered_on_a_requirement_nobody_stated(self):
        from jobscout.matching import resolve_seniority

        verdict = resolve_seniority(self._job(), candidate_years=0.5)
        assert verdict.filtered is False

    def test_the_prose_says_five_even_though_the_parser_cannot_reach_it(self):
        """Pins the remaining gap so the eventual fix has a target.

        `5+ years` appears twice; the nearest experience term is 78 chars away
        and PROXIMITY_WINDOW is 60. When that window changes, this test tells
        you whether job 98 started resolving.
        """
        from jobscout.profiles import PROXIMITY_WINDOW, find_required_years

        assert "5+ years" in self.TEXT
        found = find_required_years(self.TEXT, "fr")
        assert found is None, (
            "job 98 now resolves - if PROXIMITY_WINDOW was widened past 78 "
            f"(currently {PROXIMITY_WINDOW}), update this test to expect 5.0")
