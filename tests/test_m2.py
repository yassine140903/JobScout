"""M2 tests: CV ingestion, extraction, profile storage, coexistence."""

from pathlib import Path
import json
import tempfile

from jobscout.db import init_db, migrate_m2, upsert_profile, get_profile, get_all_profiles
from jobscout.profiles import extract_text, RuleBasedExtractor


def _temp_db():
    return Path(tempfile.mktemp(suffix=".db"))


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def test_migrate_m2_adds_columns():
    db = _temp_db()
    conn = init_db(db)
    migrate_m2(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(profiles)").fetchall()}
    expected = {"raw_text", "skills", "domains", "seniority", "languages",
                "target_locations", "company_types", "position_types"}
    assert expected.issubset(columns)

    conn.close()
    db.unlink()


def test_migrate_m2_is_idempotent():
    db = _temp_db()
    conn = init_db(db)
    migrate_m2(conn)
    migrate_m2(conn)  # second call should not raise

    columns = {row[1] for row in conn.execute("PRAGMA table_info(profiles)").fetchall()}
    assert "skills" in columns

    conn.close()
    db.unlink()


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def test_extract_pdf(tmp_path):
    """Extract text from a minimal PDF."""
    import pdfplumber
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Python developer with 5 years experience")
    c.save()

    text = extract_text(pdf_path)
    assert "python" in text.lower()
    assert "5 years" in text.lower() or "5years" in text.lower()


def test_extract_docx(tmp_path):
    """Extract text from a minimal DOCX."""
    from docx import Document

    docx_path = tmp_path / "test.docx"
    doc = Document()
    doc.add_paragraph("Senior ML engineer fluent in French and English")
    doc.save(str(docx_path))

    text = extract_text(docx_path)
    assert "senior" in text.lower()
    assert "ml engineer" in text.lower()


def test_extract_unsupported_raises():
    import pytest
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(Path("resume.txt"))


# ---------------------------------------------------------------------------
# Rule-based extractor
# ---------------------------------------------------------------------------

def test_extractor_skills():
    extractor = RuleBasedExtractor()
    text = "Experienced with Python, Docker, Kubernetes, and PostgreSQL. Built CI/CD pipelines."
    result = extractor.extract(text)
    assert "python" in result["skills"]
    assert "docker" in result["skills"]
    assert "kubernetes" in result["skills"]
    assert "postgresql" in result["skills"]


def test_extractor_case_sensitive_r():
    extractor = RuleBasedExtractor()
    text = "Statistical analysis using R and Python for data visualization"
    result = extractor.extract(text)
    assert "r" in result["skills"]
    assert "python" in result["skills"]


def test_extractor_r_no_false_positive():
    extractor = RuleBasedExtractor()
    text = "Worked in R&D department on various projects"
    result = extractor.extract(text)
    assert "r" not in result["skills"]


def test_extractor_domains():
    extractor = RuleBasedExtractor()
    text = "Focused on machine learning and MLOps for healthcare applications"
    result = extractor.extract(text)
    assert "machine learning" in result["domains"]
    assert "mlops" in result["domains"]
    assert "healthcare" in result["domains"]


def test_extractor_domain_aliases():
    extractor = RuleBasedExtractor()
    text = "Built ML pipelines and DL models for NLP tasks"
    result = extractor.extract(text)
    assert "machine learning" in result["domains"]
    assert "deep learning" in result["domains"]
    assert "natural language processing" in result["domains"]


def test_extractor_education_not_domain():
    extractor = RuleBasedExtractor()
    text = "Education\nMaster in Computer Science\nWorked on ML systems"
    result = extractor.extract(text)
    assert "education" not in result["domains"]


def test_extractor_seniority_explicit():
    extractor = RuleBasedExtractor()
    assert extractor.extract("Senior software engineer")["seniority"] == "senior"
    assert extractor.extract("Team lead for backend")["seniority"] == "lead"
    assert extractor.extract("Junior developer role")["seniority"] == "junior"


def test_extractor_seniority_years_fallback():
    extractor = RuleBasedExtractor()
    assert extractor.extract("7 years of experience in backend")["seniority"] == "senior"
    assert extractor.extract("2 ans d'expérience")["seniority"] == "junior"


def test_extractor_seniority_default_junior():
    extractor = RuleBasedExtractor()
    result = extractor.extract("Built web applications with React")
    assert result["seniority"] == "junior"


def test_extractor_intern_maps_to_junior():
    extractor = RuleBasedExtractor()
    result = extractor.extract("Looking for an internship in data science")
    assert result["seniority"] == "junior"


def test_extractor_languages():
    extractor = RuleBasedExtractor()
    text = "Languages: English (fluent), French (native), Arabic"
    result = extractor.extract(text)
    assert "English" in result["languages"]
    assert "French" in result["languages"]
    assert "Arabic" in result["languages"]


def test_extractor_languages_french_names():
    extractor = RuleBasedExtractor()
    text = "Langues: anglais (courant), français (natif)"
    result = extractor.extract(text)
    assert "English" in result["languages"]
    assert "French" in result["languages"]


# ---------------------------------------------------------------------------
# Profile storage
# ---------------------------------------------------------------------------

def test_upsert_and_get_profile():
    db = _temp_db()
    conn = init_db(db)
    migrate_m2(conn)

    profile = {
        "name": "test-profile",
        "raw_text": "Some CV text",
        "skills": ["python", "docker"],
        "domains": ["machine learning"],
        "seniority": "junior",
        "languages": ["English", "French"],
        "target_locations": ["Berlin", "Paris"],
        "company_types": ["startup"],
        "position_types": ["job"],
    }
    profile_id = upsert_profile(conn, profile)
    assert profile_id > 0

    stored = get_profile(conn, "test-profile")
    assert stored is not None
    assert stored["seniority"] == "junior"
    assert json.loads(stored["skills"]) == ["python", "docker"]
    assert json.loads(stored["target_locations"]) == ["Berlin", "Paris"]

    conn.close()
    db.unlink()


def test_upsert_updates_existing():
    db = _temp_db()
    conn = init_db(db)
    migrate_m2(conn)

    profile = {
        "name": "my-profile",
        "raw_text": "Original text",
        "skills": ["python"],
        "domains": [],
        "seniority": "junior",
        "languages": ["English"],
        "target_locations": ["Europe"],
        "company_types": [],
        "position_types": ["job"],
    }
    id1 = upsert_profile(conn, profile)

    profile["skills"] = ["python", "docker", "kubernetes"]
    profile["seniority"] = "mid"
    id2 = upsert_profile(conn, profile)

    assert id1 == id2  # same profile, not a new row
    stored = get_profile(conn, "my-profile")
    assert json.loads(stored["skills"]) == ["python", "docker", "kubernetes"]
    assert stored["seniority"] == "mid"

    conn.close()
    db.unlink()


def test_two_profiles_coexist():
    db = _temp_db()
    conn = init_db(db)
    migrate_m2(conn)

    for name, locs in [("profile-a", ["Berlin"]), ("profile-b", ["Paris"])]:
        upsert_profile(conn, {
            "name": name,
            "raw_text": "Same CV text",
            "skills": ["python"],
            "domains": ["machine learning"],
            "seniority": "junior",
            "languages": ["English"],
            "target_locations": locs,
            "company_types": [],
            "position_types": ["job"],
        })

    profiles = get_all_profiles(conn)
    assert len(profiles) == 2
    names = {p["name"] for p in profiles}
    assert names == {"profile-a", "profile-b"}

    conn.close()
    db.unlink()