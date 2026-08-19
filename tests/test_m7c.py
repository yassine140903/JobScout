"""M7c: the experience requirement read out of the description prose.

The structured field is null for postings that state a requirement outright,
so the fallback chain gained a layer between the API field and the title
guess. These cover the parser's patterns, the two context guards that keep it
from reading company anniversaries and degree lengths as job requirements, the
new chain order, and the filtering rule that treats prose as a stated
requirement while a title guess stays a guess.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from jobscout.db import (
    init_db, migrate_m2, migrate_m3, migrate_m4, migrate_m5, migrate_m6,
    migrate_m7b, migrate_m7d, upsert_profile,
)
from jobscout.matching import (
    BEYOND_GATE_MIN,
    STATED_SOURCES,
    resolve_seniority,
    run_matching,
    score_job,
)
from jobscout.profiles import find_required_years, parse_required_years
from jobscout.sources import RawPosting, _insert_job, normalize


@pytest.fixture
def db(tmp_path):
    conn = init_db(tmp_path / "t.db")
    for migrate in (migrate_m2, migrate_m3, migrate_m4, migrate_m5,
                    migrate_m6, migrate_m7b, migrate_m7d):
        migrate(conn)
    yield conn
    conn.close()


class FakeEmbedder:
    """Every text embeds identically, so only the seniority facet moves."""

    def embed(self, text, is_query=False):
        vec = np.ones(768, dtype=np.float32)
        return vec / np.linalg.norm(vec)


def job(**overrides):
    """A job row as a plain dict, which resolve_seniority reads like a Row."""
    base = {
        "required_years_min": None,
        "title": "Data Engineer",
        "description": None,
        "language": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Step 1 — the patterns, per language
# ---------------------------------------------------------------------------

class TestFrenchPatterns:
    @pytest.mark.parametrize("text", [
        "Vous disposez d'au moins 3 ans d'expérience en Machine Learning.",
        "3 ans d'expérience",
        "Nous demandons minimum 3 ans d'expérience.",
        "Une expérience de 3 ans est requise.",
        "3+ ans d'expérience",
        "Une expérience de minimum 3 années",
        "Vous justifiez d'une expérience de 3 ans minimum.",
    ])
    def test_reads_three_years(self, text):
        assert parse_required_years(text, "fr") == 3.0

    def test_a_range_reports_its_floor(self):
        assert parse_required_years("Une expérience de 3 à 5 ans", "fr") == 3.0

    def test_a_plus_after_a_range_does_not_restart_it(self):
        """"3 à 7+ ans" states a floor of 3, not of 7."""
        assert parse_required_years("3 à 7+ ans d'expérience", "fr") == 3.0

    def test_spelled_out_numbers(self):
        assert parse_required_years("Vous avez trois ans d'expérience.", "fr") == 3.0
        assert parse_required_years("un an d'expérience minimum", "fr") == 1.0

    def test_half_years_survive_the_comma(self):
        assert parse_required_years("1,5 ans d'expérience", "fr") == 1.5


class TestEnglishPatterns:
    @pytest.mark.parametrize("text", [
        "3+ years of experience",
        "at least 3 years of experience",
        "a minimum of 3 years of experience",
        "3 years experience with Python",
    ])
    def test_reads_three_years(self, text):
        assert parse_required_years(text, "en") == 3.0

    @pytest.mark.parametrize("text", [
        "3-5 years of experience",
        "3 to 5 years of experience",
    ])
    def test_a_range_reports_its_floor(self, text):
        assert parse_required_years(text, "en") == 3.0

    def test_spelled_out_numbers(self):
        assert parse_required_years("three years of experience", "en") == 3.0


class TestGermanPatterns:
    @pytest.mark.parametrize("text", [
        "mindestens 3 Jahre Berufserfahrung",
        "3 Jahre Berufserfahrung",
        "3 Jahren Berufserfahrung in der Entwicklung",
    ])
    def test_reads_three_years(self, text):
        assert parse_required_years(text, "de") == 3.0

    def test_a_range_reports_its_floor(self):
        assert parse_required_years("3-5 Jahre Erfahrung", "de") == 3.0

    def test_spelled_out_numbers(self):
        assert parse_required_years("drei Jahre Berufserfahrung", "de") == 3.0


class TestLanguageHint:
    def test_the_hint_is_not_required(self):
        assert parse_required_years("3 ans d'expérience") == 3.0
        assert parse_required_years("3 years of experience") == 3.0
        assert parse_required_years("3 Jahre Berufserfahrung") == 3.0

    def test_a_wrong_hint_still_finds_the_requirement(self):
        """The hint orders the scan; it never suppresses another language."""
        assert parse_required_years("3 ans d'expérience", "de") == 3.0

    def test_the_hint_keeps_a_foreign_aside_from_lowering_the_floor(self):
        text = (
            "Vous justifiez de 5 ans d'expérience en data engineering. "
            "We also expect 2 years of experience with our internal tooling."
        )
        assert parse_required_years(text, "fr") == 5.0     # the posting's own claim
        assert parse_required_years(text) == 2.0           # unhinted: lowest wins

    def test_an_unknown_hint_falls_back_to_every_language(self):
        assert parse_required_years("3 ans d'expérience", "it") == 3.0


# ---------------------------------------------------------------------------
# Step 2 — the guards
# ---------------------------------------------------------------------------

class TestProximityGuard:
    @pytest.mark.parametrize("text", [
        "Simplon existe depuis 3 ans.",                       # company age
        "Un CDD de 2 ans est proposé.",                       # contract length
        "Le programme se déroule sur 2 ans.",                 # programme length
        "This role has been open for 3 years.",
        "Das Unternehmen besteht seit 3 Jahren.",
    ])
    def test_a_bare_number_of_years_is_not_a_requirement(self, text):
        assert parse_required_years(text) is None

    def test_an_experience_term_within_the_window_is_enough(self):
        assert parse_required_years(
            "Expérience requise sur des projets similaires : 3 ans.", "fr",
        ) == 3.0

    def test_an_experience_term_far_away_does_not_count(self):
        text = "Simplon existe depuis 3 ans." + " " * 200 + "Expérience: Python."
        assert parse_required_years(text, "fr") is None


class TestEducationExclusion:
    @pytest.mark.parametrize("text", [
        "Une formation de 3 ans",
        "Un diplôme obtenu en 5 ans d'études",
        "Un cursus de 3 ans en informatique",
        "Bac+5, soit 5 ans après le bac",
        "A degree taking 4 years to complete",
        "Ein Studium von 3 Jahren",
    ])
    def test_education_durations_are_not_experience(self, text):
        assert parse_required_years(text) is None

    def test_the_nearer_term_decides(self):
        """A posting can name a degree and a requirement in the same breath."""
        text = "Diplômé d'un Bac+5, vous avez 3 ans d'expérience en Python."
        assert parse_required_years(text, "fr") == 3.0

    def test_education_wins_a_tie(self):
        assert parse_required_years("expérience formation de 3 ans", "fr") is None


class TestImplausibleValues:
    def test_a_company_anniversary_is_not_a_requirement(self):
        assert parse_required_years(
            "Backed by over 25 years of experience, we serve clients worldwide.",
            "en",
        ) is None

    def test_it_only_caps_the_absurd(self):
        assert parse_required_years("15 years of experience required", "en") == 15.0


class TestMultipleMatches:
    def test_the_minimum_wins(self):
        text = (
            "Vous avez 5 ans d'expérience en Java. "
            "Une expérience de 2 ans en Python est également demandée."
        )
        assert parse_required_years(text, "fr") == 2.0

    def test_a_rejected_match_does_not_veto_an_accepted_one(self):
        text = (
            "Simplon existe depuis 1 an. "
            "Vous justifiez de 4 ans d'expérience en data engineering."
        )
        assert parse_required_years(text, "fr") == 4.0


class TestNoAnswerIsAnAnswer:
    @pytest.mark.parametrize("text", [None, "", "   ", "Nous recherchons un ingénieur."])
    def test_returns_none_rather_than_guessing(self, text):
        assert find_required_years(text) is None
        if text is not None:
            assert parse_required_years(text) is None


class TestEvidence:
    def test_the_snippet_shows_the_words_that_matched(self):
        found = find_required_years("Vous disposez d'au moins 3 ans d'expérience.", "fr")
        assert found.years == 3.0
        assert found.matched == "3 ans"
        assert "3 ans d'expérience" in found.snippet
        assert found.language == "fr"

    def test_the_snippet_is_a_single_line(self):
        found = find_required_years("Profil\n\n* 3 ans\nd'expérience\n", "fr")
        assert "\n" not in found.snippet


# ---------------------------------------------------------------------------
# Step 3 — the fallback chain
# ---------------------------------------------------------------------------

class TestFallbackChainOrder:
    DESCRIPTION = "Vous disposez d'au moins 3 ans d'expérience."

    def test_api_beats_description(self):
        verdict = resolve_seniority(
            job(required_years_min=5.0, description=self.DESCRIPTION), 0.0,
        )
        assert verdict.source == "api"
        assert verdict.required_years == 5.0

    def test_description_beats_title(self):
        verdict = resolve_seniority(
            job(description=self.DESCRIPTION, title="Senior ML Engineer"), 0.0,
        )
        assert verdict.source == "description"
        assert verdict.required_years == 3.0    # not the title's 5.0

    def test_title_is_used_when_the_description_says_nothing(self):
        verdict = resolve_seniority(
            job(description="Rejoignez une équipe qui construit.",
                title="Senior ML Engineer"), 0.0,
        )
        assert verdict.source == "title"
        assert verdict.required_years == 5.0

    def test_nothing_anywhere_stays_neutral(self):
        verdict = resolve_seniority(
            job(description="Rejoignez une équipe qui construit."), 0.0,
        )
        assert verdict.source == "none"
        assert verdict.required_years is None
        assert verdict.multiplier == 1.0
        assert verdict.filtered is False

    def test_a_missing_description_column_is_survivable(self):
        """resolve_seniority still takes plain dicts that predate the layer."""
        verdict = resolve_seniority({"required_years_min": None, "title": "X"}, 0.0)
        assert verdict.source == "none"

    def test_the_posting_language_is_passed_to_the_parser(self):
        verdict = resolve_seniority(
            job(
                description=(
                    "Vous justifiez de 5 ans d'expérience. "
                    "We also expect 2 years of experience with our tooling."
                ),
                language="fr",
            ),
            0.0,
        )
        assert verdict.required_years == 5.0


class TestDescriptionCarriesItsEvidence:
    def test_the_verdict_holds_the_snippet(self):
        verdict = resolve_seniority(
            job(description="Vous disposez d'au moins 3 ans d'expérience."), 0.0,
        )
        assert "3 ans d'expérience" in verdict.snippet

    def test_other_sources_carry_no_snippet(self):
        assert resolve_seniority(job(required_years_min=3.0), 0.0).snippet is None
        assert resolve_seniority(job(title="Senior Engineer"), 0.0).snippet is None

    def test_the_audit_payload_exposes_it(self):
        verdict = resolve_seniority(
            job(description="Vous disposez d'au moins 3 ans d'expérience."), 0.0,
        )
        payload = verdict.as_dict()
        assert payload["source"] == "description"
        assert "3 ans" in payload["snippet"]


# ---------------------------------------------------------------------------
# Step 4 — what may filter
# ---------------------------------------------------------------------------

class TestDescriptionFiltersAndTitleDoesNot:
    DESCRIPTION = "Vous disposez d'au moins 5 ans d'expérience."

    def test_a_requirement_stated_in_prose_filters(self):
        verdict = resolve_seniority(
            job(description=self.DESCRIPTION), 0.75, gate=2.0,
        )
        assert verdict.source == "description"
        assert verdict.filtered is True

    def test_a_title_guess_still_does_not(self):
        verdict = resolve_seniority(job(title="Senior ML Engineer"), 0.75, gate=2.0)
        assert verdict.source == "title"
        assert verdict.filtered is False

    def test_both_are_penalised_identically(self):
        """Only the filtering differs; prose and title cost the same rank."""
        prose = resolve_seniority(job(description=self.DESCRIPTION), 0.75, gate=2.0)
        title = resolve_seniority(job(title="Senior ML Engineer"), 0.75, gate=2.0)
        assert prose.multiplier == title.multiplier
        assert prose.filtered != title.filtered

    def test_prose_within_the_gate_is_not_filtered(self):
        verdict = resolve_seniority(
            job(description="Une expérience de 2 ans est demandée."), 0.75, gate=2.0,
        )
        assert verdict.source == "description"
        assert verdict.filtered is False

    def test_stated_sources_are_the_two_that_filter(self):
        assert STATED_SOURCES == frozenset({"api", "description"})

    def test_the_inferred_switch_does_not_change_prose(self):
        for flag in (False, True):
            verdict = resolve_seniority(
                job(description=self.DESCRIPTION), 0.75, gate=2.0,
                filter_on_inferred=flag,
            )
            assert verdict.filtered is True


# ---------------------------------------------------------------------------
# Step 5 — the two carried-over fixes
# ---------------------------------------------------------------------------

class TestScorePrecision:
    def _score(self, multiplier_source):
        vec = np.ones(768, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        return score_job(
            profile_skills_emb=vec, profile_domain_emb=vec,
            job_skills_emb=vec, job_domain_emb=vec,
            profile_skills=[], job_skills=[],
            seniority=multiplier_source,
        )

    def test_final_score_keeps_six_decimals(self):
        verdict = resolve_seniority(job(required_years_min=3.0), 0.75, gate=2.0)
        final = self._score(verdict)["final_score"]
        assert final == pytest.approx(verdict.multiplier, abs=1e-9)
        assert round(final, 6) == final

    def test_scores_four_decimals_apart_no_longer_tie(self):
        """Two jobs a hair apart used to land on the same rounded score."""
        near = resolve_seniority(job(required_years_min=2.74), 0.75, gate=2.0)
        far = resolve_seniority(job(required_years_min=2.75), 0.75, gate=2.0)
        assert near.multiplier != far.multiplier
        assert self._score(near)["final_score"] != self._score(far)["final_score"]


class TestSeniorityFloor:
    def test_the_floor_is_raised(self):
        assert BEYOND_GATE_MIN == 0.15

    def test_stretch_roles_still_order_among_themselves(self):
        multipliers = [
            resolve_seniority(job(required_years_min=y), 0.75, gate=2.0).multiplier
            for y in (4.0, 6.0, 10.0)
        ]
        assert multipliers == sorted(multipliers, reverse=True)
        assert min(multipliers) >= 0.15

    def test_nothing_is_scored_below_the_floor(self):
        verdict = resolve_seniority(job(required_years_min=20.0), 0.0, gate=2.0)
        assert verdict.multiplier == 0.15
        assert verdict.filtered is True


# ---------------------------------------------------------------------------
# Step 6 — end to end
# ---------------------------------------------------------------------------

class TestMatchingIntegration:
    def _seed(self, db):
        upsert_profile(db, {
            "name": "p", "raw_text": "", "skills": [], "domains": [],
            "seniority": "junior", "languages": [], "target_locations": [],
            "company_types": [], "position_types": [],
        })
        postings = [
            # api wins over the prose it contradicts
            ("Architect Cloud et Data H/F", 3.0,
             "Vous disposez d'au moins 8 ans d'expérience."),
            # prose is the only stated requirement
            ("Machine Learning Engineer — ML, Python & MLOps", None,
             "Vous disposez d'au moins 3 ans d'expérience en Machine Learning."),
            # nothing stated, title inflated
            ("Senior ML Engineer", None, "Rejoignez une équipe qui construit."),
            # nothing stated anywhere
            ("Data Engineer", None, "Rejoignez une équipe qui construit."),
        ]
        for i, (title, years, description) in enumerate(postings):
            _insert_job(db, normalize(RawPosting(
                title=title, source="wttj", url=f"https://x/{i}", source_id=str(i),
                description=description, language="fr",
                required_years_min=years,
                seniority_source="api" if years is not None else None,
            )))
        db.commit()

    def test_the_chain_runs_end_to_end(self, db):
        self._seed(db)
        results = run_matching(db, "p", embedder=FakeEmbedder(),
                               candidate_years=0.75, gate_years=2.0)
        by_title = {r["job_title"]: r["seniority"] for r in results}

        assert by_title["Architect Cloud et Data H/F"].source == "api"
        assert by_title["Architect Cloud et Data H/F"].required_years == 3.0

        prose = by_title["Machine Learning Engineer — ML, Python & MLOps"]
        assert prose.source == "description"
        assert prose.required_years == 3.0
        assert "3 ans d'expérience" in prose.snippet

        assert by_title["Senior ML Engineer"].source == "title"
        assert by_title["Data Engineer"].source == "none"

    def test_prose_filters_and_the_title_guess_does_not(self, db):
        self._seed(db)
        results = run_matching(db, "p", embedder=FakeEmbedder(),
                               candidate_years=0.75, gate_years=2.0)
        assert {r["job_title"] for r in results if r["filtered"]} == {
            "Architect Cloud et Data H/F",
            "Machine Learning Engineer — ML, Python & MLOps",
        }

    def test_the_source_is_persisted_for_inspection(self, db):
        self._seed(db)
        run_matching(db, "p", embedder=FakeEmbedder(),
                     candidate_years=0.75, gate_years=2.0)
        row = db.execute(
            "SELECT seniority_source, required_years_min FROM jobs"
            " WHERE title LIKE 'Machine Learning Engineer%'"
        ).fetchone()
        assert row["seniority_source"] == "description"
        # The column stays the source's own structured field; the description
        # layer resolves at match time and never writes back to it.
        assert row["required_years_min"] is None

    def test_the_snippet_reaches_the_stored_audit_trail(self, db):
        self._seed(db)
        run_matching(db, "p", embedder=FakeEmbedder(),
                     candidate_years=0.75, gate_years=2.0)
        facets = db.execute(
            "SELECT m.facet_scores FROM matches m JOIN jobs j ON j.id = m.job_id"
            " WHERE j.title LIKE 'Machine Learning Engineer%'"
        ).fetchone()[0]
        detail = json.loads(facets)["_seniority"]
        assert detail["source"] == "description"
        assert "3 ans" in detail["snippet"]

    def test_rerunning_is_idempotent(self, db):
        self._seed(db)
        first = run_matching(db, "p", embedder=FakeEmbedder(),
                             candidate_years=0.75, gate_years=2.0)
        second = run_matching(db, "p", embedder=FakeEmbedder(),
                              candidate_years=0.75, gate_years=2.0)
        assert [r["final_score"] for r in first] == [r["final_score"] for r in second]
        assert db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == len(first)


class TestTheMilestoneCase:
    """eXalt, Bordeaux: null structured field, a requirement stated in prose."""

    DESCRIPTION = (
        "Vous disposez d'au moins **3 ans d'expérience en Machine Learning "
        "Engineering, Data Science orientée production, MLOps ou dans un rôle "
        "similaire**.\n\nVous avez une solide maîtrise de **Python**."
    )

    def test_it_is_read_from_the_prose_and_filtered(self):
        verdict = resolve_seniority(
            job(
                title="Machine Learning Engineer — ML, Python & MLOps",
                description=self.DESCRIPTION,
                language="fr",
            ),
            candidate_years=0.75,
            gate=2.0,
        )
        assert verdict.required_years == 3.0
        assert verdict.source == "description"
        assert verdict.gap == 2.25
        assert verdict.filtered is True

    def test_markdown_emphasis_does_not_hide_the_number(self):
        assert parse_required_years(self.DESCRIPTION, "fr") == 3.0
