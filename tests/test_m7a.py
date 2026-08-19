"""M7a: HTML cleaning and post-extraction text quality guard."""

from __future__ import annotations

import pytest

from jobscout.textclean import (
    check_text_quality,
    clean_description,
    longest_alnum_run,
    tokenize,
)


# ---------------------------------------------------------------------------
# clean_description
# ---------------------------------------------------------------------------

class TestCleanDescription:
    def test_empty_and_none(self):
        assert clean_description("") == ""
        assert clean_description(None) == ""

    def test_plain_text_passthrough(self):
        text = "Python and SQL required. 5+ years experience."
        assert clean_description(text) == text

    def test_plain_multiline_text_keeps_structure(self):
        text = "Requirements:\n- Python\n- SQL"
        assert clean_description(text) == text

    def test_strips_tags_but_keeps_text(self):
        assert clean_description("<p>Hello <b>world</b></p>") == "Hello world"

    def test_drops_script_contents(self):
        out = clean_description("<div>Keep<script>var x = 1; alert('no')</script>this</div>")
        assert "var x" not in out
        assert "alert" not in out
        assert "Keep" in out and "this" in out

    def test_drops_style_contents(self):
        out = clean_description("<div>Text</div><style>.cls{color:red;font-size:10px}</style>")
        assert "color:red" not in out
        assert "font-size" not in out
        assert out.strip() == "Text"

    def test_removing_script_does_not_glue_neighbours(self):
        # The whole point of the module: never fuse two words together.
        out = clean_description("<div>Keep<script>x</script>this</div>")
        assert "Keepthis" not in out

    def test_unescapes_entities(self):
        out = clean_description("<p>Python &amp; SQL &nbsp;&#39;fun&#39;</p>")
        assert "&amp;" not in out
        assert "&#39;" not in out
        assert "Python & SQL" in out
        assert "'fun'" in out

    def test_unescapes_double_escaped_entities(self):
        # Several stored descriptions contain a literal "&amp;" - double-escaped
        # at the source. Unescaping runs to a fixpoint, so it resolves.
        assert clean_description("<p>A &amp;amp; B</p>") == "A & B"

    @pytest.mark.parametrize(
        "markup,expected",
        [
            ("<ul><li>Python</li><li>SQL</li></ul>", "Python\nSQL"),
            ("a<br>b", "a\nb"),
            ("a<br/>b", "a\nb"),
            ("<p>one</p><p>two</p>", "one\n\ntwo"),
            ("<div>one</div><div>two</div>", "one\n\ntwo"),
            ("<table><tr><td>a</td><td>b</td></tr></table>", "a\nb"),
        ],
    )
    def test_structure_becomes_newlines(self, markup, expected):
        assert clean_description(markup) == expected

    def test_does_not_flatten_to_single_line(self):
        markup = "<h2>Role</h2><p>Build things.</p><ul><li>Python</li><li>SQL</li></ul>"
        out = clean_description(markup)
        assert out.count("\n") >= 3, f"structure was flattened: {out!r}"

    def test_collapses_three_or_more_newlines_to_two(self):
        out = clean_description("<p>a</p>\n\n\n\n\n\n<p>b</p>")
        assert "\n\n\n" not in out
        assert out == "a\n\nb"

    def test_strips_trailing_whitespace_per_line(self):
        out = clean_description("<p>a   </p><p>b\t</p>")
        assert not any(line != line.rstrip() for line in out.split("\n"))

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "plain text with no markup at all",
            "<p>Simple</p>",
            "<div>Keep<script>var x=1</script>this</div>",
            "<ul><li>Python</li><li>SQL</li></ul><p>Apply</p>",
            "<p>Python &amp;amp; SQL</p>",
            "<p>a</p>\n\n\n\n<p>b</p>",
            '<span style="color:#000">Vous</span> avez <b>5 ans</b>',
            "<table><tr><td>x</td></tr></table>",
        ],
    )
    def test_idempotent(self, raw):
        once = clean_description(raw)
        assert clean_description(once) == once

    def test_real_world_styled_markup(self):
        # Shape taken from a stored posting: nested spans carrying font/colour CSS.
        markup = (
            '<p><span style="font-size:24px;color:#000000;">'
            "<strong>Votre profil</strong></span></p>\n"
            '<p><span style="font-family:\'HelveticaNeueLT Std\';">'
            "Vous avez 5 ans d&#39;exp&eacute;rience</span></p>"
        )
        out = clean_description(markup)
        assert "style=" not in out
        assert "color:#000000" not in out
        assert "span" not in out
        assert "Votre profil" in out
        assert "Vous avez 5 ans d'expérience" in out
        assert clean_description(out) == out


# ---------------------------------------------------------------------------
# Text quality guard
# ---------------------------------------------------------------------------

class TestLongestAlnumRun:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("Python", 6),
            ("Snowflake/Databricks/BigQuery", 10),
            ("informatiques/technologiques", 14),
            ("across20containerizedservices", 29),
            ("", 0),
            ("---", 0),
        ],
    )
    def test_runs(self, token, expected):
        assert longest_alnum_run(token) == expected


class TestTokenize:
    def test_trims_edge_punctuation(self):
        assert tokenize("Hello, (world)!") == ["Hello", "world"]

    def test_empty(self):
        assert tokenize("   ") == []


class TestCheckTextQuality:
    # --- negative cases: clean text must pass ---

    def test_clean_english_passes(self):
        text = (
            "We are looking for a senior data engineer with strong Python and "
            "SQL skills. You will build and maintain data pipelines, work with "
            "our analytics team, and help design our cloud infrastructure on AWS. "
            "Experience with Docker and Kubernetes is a plus. We offer a flexible "
            "schedule and a competitive salary for the right candidate today."
        )
        assert check_text_quality(text).ok

    def test_clean_french_passes(self):
        text = (
            "Nous recherchons un ingenieur data confirme pour rejoindre notre "
            "equipe. Vous serez en charge de la conception et de la maintenance "
            "des pipelines de donnees, ainsi que du suivi de la qualite. Une "
            "bonne maitrise de Python et de SQL est demandee pour ce poste."
        )
        assert check_text_quality(text).ok

    def test_urls_do_not_trip_the_guard(self):
        text = (
            "Apply through our careers page at "
            "https://www.example.com/careers/very/long/path/to/a/posting?ref=abcdefghijklmnop "
            "or send your CV to recruitment.department@example-company.com today. "
            "We review every application carefully and reply within two weeks."
        )
        assert check_text_quality(text).ok

    def test_slash_joined_compounds_do_not_trip_the_guard(self):
        text = (
            "You will work with Snowflake/Databricks/BigQuery and other modern "
            "data platforms. Experience with informatiques/technologiques "
            "environments is welcome. We value curiosity and a pragmatic "
            "approach to solving hard engineering problems every single day."
        )
        assert check_text_quality(text).ok

    # --- positive cases: mangled text must be flagged ---

    def test_glued_words_flagged(self):
        text = (
            "BuiltanSSISETLpipelineconsolidatingmultipleSQLServersources "
            "Deliveredarealtimebehavioralanomalydetectionplatformforbranchoperations "
            "PostgreSQLasdurablestoreacrosscontainerizedservices"
        )
        report = check_text_quality(text)
        assert not report.ok
        assert report.long_tokens

    def test_letters_fused_with_digits_flagged(self):
        text = " ".join(f"across{i}containerizedservices" for i in range(30))
        report = check_text_quality(text)
        assert not report.ok
        assert report.alnum_mixed_ratio > 0.15

    def test_high_mean_token_length_flagged(self):
        text = " ".join(["averyverylongfusedtokenindeed"] * 30)
        report = check_text_quality(text)
        assert not report.ok
        assert any("mean token length" in r for r in report.reasons)

    def test_empty_text_flagged(self):
        report = check_text_quality("   ")
        assert not report.ok
        assert report.n_tokens == 0

    def test_reasons_are_populated_and_readable(self):
        report = check_text_quality("PostgreSQLasdurablestoreacross20containerizedservices")
        assert not report.ok
        assert report.reasons
        assert isinstance(report.summary(), str)

    def test_thresholds_are_configurable(self):
        text = " ".join(["Berufsunfaehigkeitsversicherung"] * 30)
        assert not check_text_quality(text).ok
        assert check_text_quality(
            text, max_token_length=40, max_mean_token_length=40.0
        ).ok

    def test_short_text_skips_ratio_checks(self):
        # Under MIN_TOKENS_FOR_RATIOS the ratios are noise and must not fire.
        report = check_text_quality("Python3 SQL2")
        assert not any("ratio" in r for r in report.reasons)
        assert not any("mean token length" in r for r in report.reasons)


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------

class TestNormalizeCleansDescription:
    def test_normalize_stores_clean_text(self):
        from jobscout.sources import RawPosting, normalize

        raw = RawPosting(
            title="Data Engineer",
            source="test",
            company="ACME",
            description="<p>Build <b>pipelines</b></p><script>x=1</script><ul><li>Python</li></ul>",
        )
        job = normalize(raw)
        assert "<p>" not in job["description"]
        assert "script" not in job["description"]
        assert "x=1" not in job["description"]
        assert "Python" in job["description"]
        assert "\n" in job["description"], "structure was flattened"

    def test_normalize_keeps_none_description_none(self):
        from jobscout.sources import RawPosting, normalize

        job = normalize(RawPosting(title="T", source="s", description=None))
        assert job["description"] is None
