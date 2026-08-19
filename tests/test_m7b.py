"""M7b: seniority as years.

Structured capture from WTTJ, the schema, CV date-range summing, and the
asymmetric years comparison that replaced the bucket ladder.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from jobscout.db import (
    init_db, migrate_m2, migrate_m3, migrate_m4, migrate_m5, migrate_m6,
    migrate_m7b, migrate_m7d,
)
from jobscout.matching import (
    DEFAULT_GATE_YEARS,
    infer_years_from_title,
    resolve_seniority,
    score_seniority_years,
)
from jobscout.profiles import (
    experience_section,
    parse_experience_years,
    resolve_candidate_years,
)
from jobscout.sources import RawPosting, normalize
from jobscout.sources.wttj import WTTJAdapter

TODAY = date(2026, 8, 1)


@pytest.fixture
def db(tmp_path):
    conn = init_db(tmp_path / "t.db")
    for migrate in (migrate_m2, migrate_m3, migrate_m4, migrate_m5, migrate_m6,
                    migrate_m7b, migrate_m7d):
        migrate(conn)
    yield conn
    conn.close()


def hit(**overrides):
    """A minimally viable WTTJ hit, shaped like the real API response."""
    base = {
        "name": "Architect Cloud et Data H/F",
        "objectID": "GCA_b51lqLo",
        "slug": "architect-cloud-et-devops-h-f_montrouge",
        "organization": {"name": "Crédit Agricole", "slug": "groupe-credit-agricole"},
        "office": {"city": "Montrouge", "country_code": "FR"},
        "language": "fr",
        "profile": "Architecture cloud et data.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Step 1 — capture of the structured fields
# ---------------------------------------------------------------------------

class TestExperienceCapture:
    def test_captures_years_when_flag_true(self):
        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=True, experience_level_minimum=3)
        )
        assert posting.required_years_min == 3.0
        assert isinstance(posting.required_years_min, float)
        assert posting.seniority_source == "api"

    def test_fractional_half_year_survives(self):
        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=True, experience_level_minimum=0.5)
        )
        assert posting.required_years_min == 0.5

    def test_zero_years_is_a_real_requirement_not_a_missing_one(self):
        """0 means 'entry level', which is data. It must not read as absent."""
        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=True, experience_level_minimum=0)
        )
        assert posting.required_years_min == 0.0
        assert posting.seniority_source == "api"

    def test_flag_false_yields_null_not_zero(self):
        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=False, experience_level_minimum=0)
        )
        assert posting.required_years_min is None
        assert posting.seniority_source is None

    def test_flag_absent_yields_null(self):
        posting = WTTJAdapter._hit_to_posting(hit())
        assert posting.required_years_min is None
        assert posting.seniority_source is None

    def test_flag_true_but_value_null_yields_null(self):
        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=True, experience_level_minimum=None)
        )
        assert posting.required_years_min is None

    def test_unparseable_value_yields_null_not_a_crash(self):
        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=True, experience_level_minimum="3_TO_5_YEARS")
        )
        assert posting.required_years_min is None

    def test_seniority_bucket_is_no_longer_guessed(self):
        """The dead enum table is gone; nothing fabricates a bucket here."""
        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=True, experience_level_minimum=5)
        )
        assert posting.seniority is None

    def test_map_seniority_is_removed(self):
        import jobscout.sources.wttj as wttj

        assert not hasattr(wttj, "_map_seniority")


class TestEducationAndSalaryCapture:
    def test_education_level_guarded_by_flag(self):
        assert WTTJAdapter._hit_to_posting(
            hit(has_education_level=True, education_level="BAC_5")
        ).education_level == "BAC_5"
        assert WTTJAdapter._hit_to_posting(
            hit(has_education_level=False, education_level="BAC_5")
        ).education_level is None

    def test_salary_guarded_by_flag_and_carries_currency(self):
        posting = WTTJAdapter._hit_to_posting(
            hit(
                has_salary_yearly_minimum=True,
                salary_yearly_minimum=45000,
                salary_currency="EUR",
            )
        )
        assert posting.salary_yearly_min == 45000
        assert posting.salary_currency == "EUR"

    def test_currency_dropped_when_no_amount(self):
        posting = WTTJAdapter._hit_to_posting(
            hit(
                has_salary_yearly_minimum=False,
                salary_yearly_minimum=45000,
                salary_currency="EUR",
            )
        )
        assert posting.salary_yearly_min is None
        assert posting.salary_currency is None


# ---------------------------------------------------------------------------
# Step 2 — schema and the canonical model
# ---------------------------------------------------------------------------

class TestMigration:
    M7B_COLUMNS = {
        "required_years_min", "education_level",
        "salary_yearly_min", "salary_currency", "seniority_source",
    }

    def test_adds_columns(self, db):
        cols = {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
        assert self.M7B_COLUMNS <= cols

    def test_required_years_min_is_real_not_integer(self, db):
        types = {r[1]: r[2] for r in db.execute("PRAGMA table_info(jobs)")}
        assert types["required_years_min"] == "REAL"
        assert types["salary_yearly_min"] == "INTEGER"

    def test_is_idempotent(self, db):
        migrate_m7b(db)
        migrate_m7b(db)  # must not raise
        cols = {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
        assert self.M7B_COLUMNS <= cols

    def test_fractional_years_round_trip_through_sqlite(self, db):
        from jobscout.sources import _insert_job

        row = normalize(RawPosting(
            title="Half-year job", source="wttj", url="https://x/1",
            source_id="1", required_years_min=0.5, seniority_source="api",
        ))
        _insert_job(db, row)
        db.commit()
        stored = db.execute("SELECT required_years_min FROM jobs").fetchone()[0]
        assert stored == 0.5  # would truncate to 0 under an INTEGER column

    def test_adapters_without_the_data_store_null(self, db):
        from jobscout.sources import _insert_job

        row = normalize(RawPosting(
            title="EURES job", source="eures", url="https://x/2", source_id="2",
        ))
        _insert_job(db, row)
        db.commit()
        stored = db.execute(
            "SELECT required_years_min, education_level, salary_yearly_min,"
            " salary_currency, seniority_source FROM jobs"
        ).fetchone()
        assert list(stored) == [None, None, None, None, None]


class TestNormalizeCarriesTheFields:
    def test_normalize_passes_m7b_fields_through(self):
        row = normalize(RawPosting(
            title="t", source="wttj", required_years_min=2.0,
            education_level="BAC_5", salary_yearly_min=45000,
            salary_currency="EUR", seniority_source="api",
        ))
        assert row["required_years_min"] == 2.0
        assert row["education_level"] == "BAC_5"
        assert row["salary_yearly_min"] == 45000
        assert row["salary_currency"] == "EUR"
        assert row["seniority_source"] == "api"

    def test_the_milestone_job_end_to_end(self, db):
        """The Crédit Agricole posting that started this: 3 years, from the API."""
        from jobscout.sources import _insert_job

        posting = WTTJAdapter._hit_to_posting(
            hit(has_experience_level_minimum=True, experience_level_minimum=3)
        )
        _insert_job(db, normalize(posting))
        db.commit()
        row = db.execute(
            "SELECT title, company, required_years_min, seniority_source FROM jobs"
        ).fetchone()
        assert row["title"] == "Architect Cloud et Data H/F"
        assert row["company"] == "Crédit Agricole"
        assert row["required_years_min"] == 3.0
        assert row["seniority_source"] == "api"


class TestProfileMigration:
    def test_adds_candidate_years_columns(self, db):
        cols = {r[1]: r[2] for r in db.execute("PRAGMA table_info(profiles)")}
        assert cols["candidate_years"] == "REAL"
        assert cols["candidate_years_parsed"] == "REAL"

    def test_override_survives_reingesting_the_cv(self, db):
        """A hand-set candidate_years must not be clobbered by re-upserting."""
        from jobscout.db import get_profile, upsert_profile

        upsert_profile(db, {"name": "p", "raw_text": "v1", "skills": [],
                            "domains": [], "seniority": "junior", "languages": [],
                            "target_locations": [], "company_types": [],
                            "position_types": [], "candidate_years_parsed": 0.1})
        db.execute("UPDATE profiles SET candidate_years = 4.0 WHERE name = 'p'")
        db.commit()

        upsert_profile(db, {"name": "p", "raw_text": "v2", "skills": [],
                            "domains": [], "seniority": "senior", "languages": [],
                            "target_locations": [], "company_types": [],
                            "position_types": [], "candidate_years_parsed": 0.9})

        row = get_profile(db, "p")
        assert row["candidate_years"] == 4.0      # override preserved
        assert row["candidate_years_parsed"] == 0.9  # suggestion refreshed
        assert row["raw_text"] == "v2"


# ---------------------------------------------------------------------------
# Step 3 — candidate years from CV date ranges (advisory)
# ---------------------------------------------------------------------------

def cv(experience_body: str, extra: str = "") -> str:
    return f"Professional Experience\n{experience_body}\n{extra}"


class TestExperienceSection:
    def test_education_dates_are_not_experience(self):
        text = ("Professional Experience\nDev 2024 - 2025\n"
                "Education\nEngineering Degree 2022 - 2027\n")
        assert "2024 - 2025" in experience_section(text)
        assert "2022 - 2027" not in experience_section(text)

    def test_no_experience_header_counts_nothing(self):
        """Better to suggest zero than to sum every date on the page."""
        text = "Education\nDegree 2018 - 2022\nSkills\nPython\n"
        assert experience_section(text).strip() == ""
        assert parse_experience_years(text, today=TODAY).years == 0.0

    def test_french_header_recognised(self):
        text = "Expérience professionnelle\nDev 2019 - 2022\nFormation\n2015 - 2019\n"
        assert parse_experience_years(text, today=TODAY).years == 3.0


class TestDateRangeSumming:
    @pytest.mark.parametrize("body,expected", [
        ("Dev 2021-2024", 3.0),           # year-only endpoints: Jan to Jan
        ("Dev 2021 - 2024", 3.0),
        ("Dev 2021 – 2024", 3.0),         # en dash
        ("Dev 2021 — 2024", 3.0),         # em dash
        ("Dev jan 2021 - mar 2023", 2.25),   # end month is inclusive
        ("Dev janvier 2021 - mars 2023", 2.25),
        ("Dev Jun 2026 - Jul 2026", 0.17),
    ])
    def test_formats(self, body, expected):
        assert parse_experience_years(cv(body), today=TODAY).years == expected

    @pytest.mark.parametrize("body", [
        "Dev 2021 - present",
        "Dev 2021 – Present",
        "Dev 2021 - aujourd'hui",
        "Dev depuis 2021",
        "Dev since 2021",
    ])
    def test_open_ended_runs_to_today(self, body):
        # Jan 2021 through Aug 2026 inclusive = 68 months = 5.67 years
        assert parse_experience_years(cv(body), today=TODAY).years == 5.67

    def test_overlapping_roles_are_not_double_counted(self):
        result = parse_experience_years(
            cv("Role A 2020 - 2023\nRole B 2021 - 2024"), today=TODAY,
        )
        assert result.years == 4.0      # union 2020-2024, not 3 + 3
        assert len(result.ranges) == 2  # both were seen

    def test_fully_contained_range_adds_nothing(self):
        result = parse_experience_years(
            cv("Role A 2018 - 2024\nRole B 2020 - 2022"), today=TODAY,
        )
        assert result.years == 6.0

    def test_touching_ranges_merge_without_gap(self):
        assert parse_experience_years(
            cv("A 2018 - 2020\nB 2020 - 2022"), today=TODAY,
        ).years == 4.0

    def test_disjoint_ranges_sum(self):
        assert parse_experience_years(
            cv("A 2016 - 2018\nB 2020 - 2022"), today=TODAY,
        ).years == 4.0

    def test_open_ended_overlapping_a_closed_range(self):
        result = parse_experience_years(
            cv("A 2020 - 2023\nB depuis 2022"), today=TODAY,
        )
        assert result.years == 6.67   # Jan 2020 through Aug 2026, counted once

    def test_future_dates_are_clamped_to_today(self):
        assert parse_experience_years(
            cv("Future role 2030 - 2035"), today=TODAY,
        ).years == 0.0

    def test_end_before_start_is_discarded(self):
        assert parse_experience_years(cv("Typo 2024 - 2021"), today=TODAY).years == 0.0

    def test_unparseable_dates_are_surfaced_not_silently_dropped(self):
        result = parse_experience_years(
            cv("BI Engineer Intern Summer 2024 & Summer 2025"), today=TODAY,
        )
        assert result.years == 0.0
        assert any("Summer 2024" in line for line in result.ignored)

    def test_summary_mentions_both_counted_and_ignored(self):
        result = parse_experience_years(
            cv("A 2021 - 2023\nB Summer 2024"), today=TODAY,
        )
        assert "1.00 years" not in result.summary()  # 2 years counted
        assert "2.00 years" in result.summary()
        assert "not parsed" in result.summary()


class TestResolveCandidateYears:
    def test_config_wins(self):
        profile = {"candidate_years": 7.0}
        assert resolve_candidate_years(profile, {"profile": {"candidate_years": 2.0}}) \
            == (2.0, "config")

    def test_profile_record_used_when_config_silent(self):
        assert resolve_candidate_years({"candidate_years": 7.0}, {}) == (7.0, "profile")
        assert resolve_candidate_years(
            {"candidate_years": 7.0}, {"profile": {"candidate_years": None}}
        ) == (7.0, "profile")

    def test_unset_returns_none(self):
        assert resolve_candidate_years({"candidate_years": None}, {}) == (None, "unset")
        assert resolve_candidate_years(None, None) == (None, "unset")

    def test_zero_is_a_real_answer_not_an_absence(self):
        assert resolve_candidate_years({}, {"profile": {"candidate_years": 0.0}}) \
            == (0.0, "config")

    def test_parsed_suggestion_is_never_used_automatically(self):
        """The whole point of 'advisory': it must not leak into scoring."""
        profile = {"candidate_years": None, "candidate_years_parsed": 5.0}
        assert resolve_candidate_years(profile, {}) == (None, "unset")


# ---------------------------------------------------------------------------
# Step 4 — the comparison
# ---------------------------------------------------------------------------

class TestGapScoring:
    def test_exact_match_is_full_score(self):
        assert score_seniority_years(3.0, 3.0) == (1.0, False)

    def test_over_qualified_within_grace_is_full_score(self):
        assert score_seniority_years(1.0, 3.0) == (1.0, False)

    def test_over_qualified_decays_gently_and_never_filters(self):
        mult, filtered = score_seniority_years(0.0, 10.0)
        assert filtered is False
        assert mult >= 0.85          # a token penalty at ten years over
        assert mult < 1.0

    def test_over_qualified_has_a_floor(self):
        mult, filtered = score_seniority_years(0.0, 100.0)
        assert filtered is False
        assert mult == 0.85

    def test_near_miss_is_mild(self):
        mult, filtered = score_seniority_years(1.0, 0.0)   # gap 1.0
        assert filtered is False
        assert mult == 0.75

    def test_half_year_short_is_milder_still(self):
        mult, _ = score_seniority_years(0.5, 0.0)
        assert mult == pytest.approx(0.875)

    def test_under_decays_steeply_towards_the_gate(self):
        at_gate, filtered = score_seniority_years(2.0, 0.0, gate=2.0)
        assert filtered is False
        assert at_gate == 0.25       # far below the 0.75 near-miss

    def test_beyond_the_gate_is_filtered(self):
        mult, filtered = score_seniority_years(3.0, 0.0, gate=2.0)
        assert filtered is True
        assert mult < 0.25

    def test_asymmetry_is_the_point(self):
        """Two years under is disqualifying; two years over costs nothing."""
        under, under_filtered = score_seniority_years(5.0, 0.0, gate=2.0)
        over, over_filtered = score_seniority_years(0.0, 5.0, gate=2.0)
        assert under_filtered is True and over_filtered is False
        assert over > 0.9 > under

    def test_gate_is_configurable(self):
        assert score_seniority_years(3.0, 0.0, gate=2.0)[1] is True
        assert score_seniority_years(3.0, 0.0, gate=5.0)[1] is False

    @pytest.mark.parametrize("required,candidate", [
        (None, 0.0), (3.0, None), (None, None),
    ])
    def test_missing_either_side_is_neutral_and_unfiltered(self, required, candidate):
        assert score_seniority_years(required, candidate) == (1.0, False)

    def test_multiplier_is_monotonic_in_the_gap(self):
        gaps = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        scores = [score_seniority_years(g, 0.0, gate=2.0)[0] for g in gaps]
        assert scores == sorted(scores, reverse=True)


class TestTitleInference:
    @pytest.mark.parametrize("title", [
        "Junior Data Engineer", "Data Engineer Jr", "Stagiaire Data",
        "Internship - Machine Learning", "Alternance Data Analyst",
        "Praktikum Data Science", "Werkstudent Data", "Graduate Software Engineer",
    ])
    def test_entry_titles(self, title):
        assert infer_years_from_title(title) == 0.0

    @pytest.mark.parametrize("title", [
        "Senior ML Engineer", "Lead Data Engineer", "Principal Engineer",
        "Staff Software Engineer", "Architect Cloud et Data H/F",
        "Data Engineer Confirmé", "Ingénieur expérimenté",
        "Leitender Entwickler", "Head of Data",
    ])
    def test_senior_titles(self, title):
        assert infer_years_from_title(title) == 5.0

    @pytest.mark.parametrize("title", [
        "Data Engineer", "Machine Learning Engineer", "Développeur Python", "",
    ])
    def test_neutral_titles_infer_nothing(self, title):
        assert infer_years_from_title(title) is None

    def test_none_title(self):
        assert infer_years_from_title(None) is None

    def test_entry_wins_over_senior_wording(self):
        """'Stagiaire Data Architect' is an internship, not an architect role."""
        assert infer_years_from_title("Stagiaire Data Architect") == 0.0


class TestNullFallthrough:
    def test_api_value_wins(self):
        verdict = resolve_seniority(
            {"required_years_min": 3.0, "title": "Junior Engineer"}, 0.0,
        )
        assert verdict.source == "api"
        assert verdict.required_years == 3.0   # not the title's 0.0

    def test_falls_through_to_title(self):
        verdict = resolve_seniority(
            {"required_years_min": None, "title": "Senior ML Engineer"}, 0.0,
        )
        assert verdict.source == "title"
        assert verdict.required_years == 5.0

    def test_falls_through_to_none(self):
        verdict = resolve_seniority(
            {"required_years_min": None, "title": "Data Engineer"}, 0.0,
        )
        assert verdict.source == "none"
        assert verdict.required_years is None

    def test_none_is_neutral_and_never_filtered(self):
        verdict = resolve_seniority(
            {"required_years_min": None, "title": "Data Engineer"}, 0.0,
        )
        assert verdict.multiplier == 1.0
        assert verdict.filtered is False

    def test_none_is_never_filtered_however_low_the_candidate_years(self):
        for years in (0.0, 0.5, 20.0):
            verdict = resolve_seniority({"required_years_min": None, "title": "X"}, years)
            assert verdict.filtered is False

    def test_zero_from_the_api_is_treated_as_unset(self):
        """Reversed deliberately: WTTJ sends 0 for "not stated".

        This test previously asserted that an API 0 is a stated requirement of
        zero years. It is not — the field carries "unknown" and "genuinely
        zero" in the same value, so reading it literally stopped the fallback
        chain and postings demanding 5+ years were labelled as stating zero.
        A 0 now falls through to the description, then the title.
        """
        verdict = resolve_seniority(
            {"required_years_min": 0.0, "title": "Senior Engineer"}, 0.0,
        )
        assert verdict.source == "title"
        assert verdict.required_years == 5.0

    def test_zero_with_nothing_else_to_go_on_reports_none(self):
        verdict = resolve_seniority(
            {"required_years_min": 0.0, "title": "Data Engineer"}, 0.0,
        )
        assert verdict.source == "none"
        assert verdict.required_years is None

    def test_gap_is_reported(self):
        verdict = resolve_seniority({"required_years_min": 3.0, "title": "X"}, 0.5)
        assert verdict.gap == 2.5
        assert verdict.candidate_years == 0.5

    def test_gap_is_none_when_candidate_years_unset(self):
        verdict = resolve_seniority({"required_years_min": 3.0, "title": "X"}, None)
        assert verdict.gap is None
        assert verdict.multiplier == 1.0
        assert verdict.filtered is False

    def test_the_milestone_case(self):
        """Architect Cloud et Data, requires 3, junior profile with 0 years."""
        verdict = resolve_seniority(
            {"required_years_min": 3.0, "title": "Architect Cloud et Data H/F"},
            candidate_years=0.0,
            gate=DEFAULT_GATE_YEARS,
        )
        assert verdict.source == "api"
        assert verdict.gap == 3.0
        assert verdict.filtered is True


class TestScoringIntegration:
    def test_run_matching_marks_and_reports_stretch_roles(self, db):
        import numpy as np
        from jobscout.db import upsert_profile
        from jobscout.matching import run_matching
        from jobscout.sources import _insert_job

        class FakeEmbedder:
            def embed(self, text, is_query=False):
                vec = np.ones(768, dtype=np.float32)
                return vec / np.linalg.norm(vec)

        upsert_profile(db, {
            "name": "p", "raw_text": "", "skills": ["python"], "domains": ["mlops"],
            "seniority": "junior", "languages": [], "target_locations": [],
            "company_types": [], "position_types": [],
        })
        for i, (title, years) in enumerate([
            ("Architect Cloud et Data H/F", 3.0),   # api, filtered
            # api 0 is the "not stated" sentinel, so this now resolves through
            # the title layer instead. Same number, honest provenance.
            ("Junior Data Engineer", 0.0),          # api 0 -> title -> 0.0
            ("Senior ML Engineer", None),           # title -> 5.0, filtered
            ("Data Engineer", None),                # none -> neutral
        ]):
            _insert_job(db, normalize(RawPosting(
                title=title, source="wttj", url=f"https://x/{i}", source_id=str(i),
                required_years_min=years,
                seniority_source="api" if years is not None else None,
            )))
        db.commit()

        results = run_matching(
            db, "p", embedder=FakeEmbedder(), candidate_years=0.0, gate_years=2.0,
        )
        by_title = {r["job_title"]: r for r in results}

        assert by_title["Architect Cloud et Data H/F"]["filtered"] is True
        assert by_title["Junior Data Engineer"]["filtered"] is False
        # Title-inferred: penalised but visible, since filter_on_inferred is off.
        assert by_title["Senior ML Engineer"]["filtered"] is False
        assert by_title["Senior ML Engineer"]["seniority_multiplier"] < 0.25
        assert by_title["Data Engineer"]["filtered"] is False

        sources = {t: r["seniority"].source for t, r in by_title.items()}
        assert sources == {
            "Architect Cloud et Data H/F": "api",
            "Junior Data Engineer": "title",
            "Senior ML Engineer": "title",
            "Data Engineer": "none",
        }

    def test_seniority_source_is_persisted_for_inspection(self, db):
        import numpy as np
        from jobscout.db import upsert_profile
        from jobscout.matching import run_matching
        from jobscout.sources import _insert_job

        class FakeEmbedder:
            def embed(self, text, is_query=False):
                vec = np.ones(768, dtype=np.float32)
                return vec / np.linalg.norm(vec)

        upsert_profile(db, {
            "name": "p", "raw_text": "", "skills": [], "domains": [],
            "seniority": "junior", "languages": [], "target_locations": [],
            "company_types": [], "position_types": [],
        })
        _insert_job(db, normalize(RawPosting(
            title="Senior ML Engineer", source="eures", url="https://x/1",
            source_id="1",
        )))
        db.commit()

        run_matching(db, "p", embedder=FakeEmbedder(), candidate_years=0.0)
        assert db.execute("SELECT seniority_source FROM jobs").fetchone()[0] == "title"

    def test_facet_scores_carry_the_audit_trail(self, db):
        import json

        import numpy as np
        from jobscout.db import upsert_profile
        from jobscout.matching import run_matching
        from jobscout.sources import _insert_job

        class FakeEmbedder:
            def embed(self, text, is_query=False):
                vec = np.ones(768, dtype=np.float32)
                return vec / np.linalg.norm(vec)

        upsert_profile(db, {
            "name": "p", "raw_text": "", "skills": [], "domains": [],
            "seniority": "junior", "languages": [], "target_locations": [],
            "company_types": [], "position_types": [],
        })
        _insert_job(db, normalize(RawPosting(
            title="Architect Cloud et Data H/F", source="wttj", url="https://x/1",
            source_id="1", required_years_min=3.0, seniority_source="api",
        )))
        db.commit()

        run_matching(db, "p", embedder=FakeEmbedder(), candidate_years=0.0,
                     gate_years=2.0)
        facets = json.loads(db.execute("SELECT facet_scores FROM matches").fetchone()[0])

        assert facets["_seniority"] == {
            "multiplier": facets["seniority"],
            "filtered": True,
            "required_years": 3.0,
            "candidate_years": 0.0,
            "gap": 3.0,
            "source": "api",
            # M7c: evidence slot, filled only when the requirement was read
            # out of the description prose.
            "snippet": None,
        }


class TestStretchQuery:
    """The web view hides stretch roles with SQL, so the expression must work."""

    STRETCH_EXPR = "COALESCE(json_extract(m.facet_scores, '$._seniority.filtered'), 0)"

    def _rows(self, db):
        return db.execute(
            f"SELECT j.title, {self.STRETCH_EXPR} AS is_stretch "
            "FROM jobs j JOIN matches m ON m.job_id = j.id ORDER BY j.title"
        ).fetchall()

    def test_flag_is_readable_from_sql(self, db):
        from jobscout.sources import _insert_job

        db.execute("INSERT INTO profiles (name) VALUES ('p')")
        for i, (title, payload) in enumerate([
            ("Filtered", '{"seniority": 0.1, "_seniority": {"filtered": true}}'),
            ("Kept", '{"seniority": 1.0, "_seniority": {"filtered": false}}'),
            ("Legacy", '{"seniority": 1.0}'),   # scored before M7b
        ]):
            _insert_job(db, normalize(RawPosting(
                title=title, source="wttj", url=f"https://x/{i}", source_id=str(i),
            )))
            db.execute(
                "INSERT INTO matches (profile_id, job_id, score, facet_scores) "
                "VALUES (1, ?, 0.5, ?)",
                (db.execute("SELECT id FROM jobs WHERE title = ?", (title,)).fetchone()[0],
                 payload),
            )
        db.commit()

        flags = {r["title"]: r["is_stretch"] for r in self._rows(db)}
        assert flags["Filtered"] == 1
        assert flags["Kept"] == 0
        # A match written before M7b has no flag at all and must not be hidden.
        assert flags["Legacy"] == 0


# ---------------------------------------------------------------------------
# Follow-up fixes: config override, inferred-filter switch, EURES company
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestConfigOverridesProfileRecord:
    def test_config_beats_the_profile_record(self):
        profile = {"candidate_years": 0.0, "candidate_years_parsed": 0.17}
        years, source = resolve_candidate_years(
            profile, {"profile": {"candidate_years": 0.75}},
        )
        assert (years, source) == (0.75, "config")

    def test_shipped_config_value_is_picked_up(self):
        """The real config.yaml, loaded the way the app loads it."""
        from jobscout.config import load_config

        config = load_config(REPO_ROOT / "config.yaml")
        assert config["profile"]["candidate_years"] == 0.75
        assert resolve_candidate_years({"candidate_years": None}, config) \
            == (0.75, "config")

    def test_config_override_survives_reingest(self, db):
        """Re-ingesting rewrites the profile row; the config value is untouched."""
        from jobscout.db import get_profile, upsert_profile

        config = {"profile": {"candidate_years": 0.75}}
        base = {"name": "p", "skills": [], "domains": [], "languages": [],
                "target_locations": [], "company_types": [], "position_types": [],
                "seniority": "junior"}

        upsert_profile(db, {**base, "raw_text": "v1", "candidate_years_parsed": 0.17})
        assert resolve_candidate_years(get_profile(db, "p"), config) == (0.75, "config")

        upsert_profile(db, {**base, "raw_text": "v2", "candidate_years_parsed": 9.9})
        assert resolve_candidate_years(get_profile(db, "p"), config) == (0.75, "config")

    def test_zero_in_config_still_wins_over_the_record(self):
        """0.0 is a value, not an absence — it must not fall through."""
        assert resolve_candidate_years(
            {"candidate_years": 5.0}, {"profile": {"candidate_years": 0.0}},
        ) == (0.0, "config")


class TestFilterOnInferred:
    JOB_TITLE = {"required_years_min": None, "title": "Senior ML Engineer"}
    JOB_API = {"required_years_min": 5.0, "title": "Senior ML Engineer"}

    def test_inferred_is_penalised_but_never_filtered(self):
        verdict = resolve_seniority(self.JOB_TITLE, 0.75, gate=2.0)
        assert verdict.source == "title"
        assert verdict.filtered is False       # visible
        assert verdict.multiplier < 0.25       # but ranked well down

    def test_stated_requirement_beyond_the_gate_still_filters(self):
        verdict = resolve_seniority(self.JOB_API, 0.75, gate=2.0)
        assert verdict.source == "api"
        assert verdict.filtered is True

    def test_inferred_and_stated_get_the_same_multiplier(self):
        """Only the filtering differs — the penalty is identical."""
        inferred = resolve_seniority(self.JOB_TITLE, 0.75, gate=2.0)
        stated = resolve_seniority(self.JOB_API, 0.75, gate=2.0)
        assert inferred.multiplier == stated.multiplier
        assert inferred.filtered != stated.filtered

    def test_switch_can_be_flipped_on(self):
        verdict = resolve_seniority(
            self.JOB_TITLE, 0.75, gate=2.0, filter_on_inferred=True,
        )
        assert verdict.filtered is True

    def test_inferred_within_the_gate_is_unaffected(self):
        verdict = resolve_seniority(
            {"required_years_min": None, "title": "Junior Engineer"}, 0.75, gate=2.0,
        )
        assert verdict.source == "title"
        assert verdict.filtered is False
        assert verdict.multiplier == 1.0

    def test_none_stays_neutral_and_unfiltered(self):
        for flag in (False, True):
            verdict = resolve_seniority(
                {"required_years_min": None, "title": "Data Engineer"}, 0.75,
                gate=2.0, filter_on_inferred=flag,
            )
            assert verdict.source == "none"
            assert verdict.filtered is False
            assert verdict.multiplier == 1.0

    def test_default_is_off(self):
        from jobscout.matching import DEFAULT_FILTER_ON_INFERRED

        assert DEFAULT_FILTER_ON_INFERRED is False
        assert resolve_seniority(self.JOB_TITLE, 0.75, gate=2.0).filtered is False

    def test_shipped_config_turns_it_off(self):
        from jobscout.config import load_config

        config = load_config(REPO_ROOT / "config.yaml")
        assert config["scoring"]["seniority"]["filter_on_inferred"] is False

    def test_run_matching_honours_the_switch(self, db):
        import numpy as np
        from jobscout.db import upsert_profile
        from jobscout.matching import run_matching
        from jobscout.sources import _insert_job

        class FakeEmbedder:
            def embed(self, text, is_query=False):
                vec = np.ones(768, dtype=np.float32)
                return vec / np.linalg.norm(vec)

        upsert_profile(db, {
            "name": "p", "raw_text": "", "skills": [], "domains": [],
            "seniority": "junior", "languages": [], "target_locations": [],
            "company_types": [], "position_types": [],
        })
        for i, (title, years) in enumerate([
            ("Architect Cloud et Data H/F", 3.0),   # api
            ("Senior ML Engineer", None),           # title -> 5.0
        ]):
            _insert_job(db, normalize(RawPosting(
                title=title, source="wttj", url=f"https://x/{i}", source_id=str(i),
                required_years_min=years,
            )))
        db.commit()

        def filtered_titles(**kwargs):
            results = run_matching(db, "p", embedder=FakeEmbedder(),
                                   candidate_years=0.75, gate_years=2.0, **kwargs)
            return {r["job_title"] for r in results if r["filtered"]}

        assert filtered_titles() == {"Architect Cloud et Data H/F"}
        assert filtered_titles(filter_on_inferred=True) == {
            "Architect Cloud et Data H/F", "Senior ML Engineer",
        }


class TestEURESCompanyName:
    def test_extracts_name_from_the_employer_object(self):
        from jobscout.sources.eures import EURESAdapter

        posting = EURESAdapter._item_to_posting({
            "title": "Data Engineer",
            "id": "abc",
            "employer": {
                "name": "KAISCHOOL", "legalID": None, "sectorCodes": [],
                "organisationSizeCode": None, "website": None,
            },
        })
        assert posting.company == "KAISCHOOL"

    def test_plain_string_employer_still_works(self):
        from jobscout.sources.eures import employer_name

        assert employer_name("ACME NV") == "ACME NV"
        assert employer_name("  ACME NV  ") == "ACME NV"

    def test_missing_employer_is_none(self):
        from jobscout.sources.eures import employer_name

        assert employer_name(None) is None
        assert employer_name({}) is None
        assert employer_name({"name": None}) is None
        assert employer_name({"name": "   "}) is None
        assert employer_name("") is None


class TestEURESCompanyRepair:
    def test_repairs_a_python_repr(self):
        from jobscout.sources.eures import repair_company_value

        stored = ("{'name': 'KAISCHOOL', 'legalID': None, "
                  "'organisationSizeCode': None, 'sectorCodes': []}")
        assert repair_company_value(stored) == ("KAISCHOOL", "repaired")

    def test_repairs_json_too(self):
        from jobscout.sources.eures import repair_company_value

        assert repair_company_value('{"name": "ACME NV", "legalID": null}') \
            == ("ACME NV", "repaired")

    def test_already_clean_is_untouched(self):
        from jobscout.sources.eures import repair_company_value

        assert repair_company_value("KAISCHOOL") == ("KAISCHOOL", "clean")
        assert repair_company_value(None) == (None, "clean")

    def test_is_idempotent(self):
        from jobscout.sources.eures import repair_company_value

        once, _ = repair_company_value("{'name': 'ACME NV', 'legalID': None}")
        twice, outcome = repair_company_value(once)
        assert twice == once == "ACME NV"
        assert outcome == "clean"

    def test_unparseable_is_left_alone(self):
        from jobscout.sources.eures import repair_company_value

        broken = "{'name': 'TRUNCATED', 'legalID': No"
        assert repair_company_value(broken) == (broken, "unparseable")

    def test_parsed_but_nameless_is_left_alone(self):
        from jobscout.sources.eures import repair_company_value

        nameless = "{'legalID': None, 'sectorCodes': []}"
        assert repair_company_value(nameless) == (nameless, "unparseable")

    def test_company_shaped_but_not_a_dict_is_clean(self):
        from jobscout.sources.eures import repair_company_value

        assert repair_company_value("Acme {Holdings}") == ("Acme {Holdings}", "clean")

    def test_repair_changes_the_dedup_hash(self):
        """company is the dedup key, so a repaired name must rehash."""
        from jobscout.sources import compute_dedup_hash
        from jobscout.sources.eures import repair_company_value

        stored = "{'name': 'ACME NV', 'legalID': None}"
        repaired, outcome = repair_company_value(stored)
        assert outcome == "repaired"
        assert compute_dedup_hash("Data Engineer", repaired) \
            != compute_dedup_hash("Data Engineer", stored)
        # and it now matches what a fresh insert of the same posting would make
        assert compute_dedup_hash("Data Engineer", repaired) \
            == compute_dedup_hash("Data Engineer", "ACME NV")
