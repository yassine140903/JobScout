"""CV text extraction and profile enrichment."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    """Extract raw text from a PDF or DOCX file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2)
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ---------------------------------------------------------------------------
# Extractor interface
# ---------------------------------------------------------------------------

class Extractor(ABC):
    """Base class for profile extraction. Subclass for LLM/ML backends."""

    @abstractmethod
    def extract(self, raw_text: str) -> dict[str, Any]:
        """Return {"skills": [...], "domains": [...], "seniority": "...", "languages": [...]}"""
        ...


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

SKILLS: dict[str, list[str]] = {
    "languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "matlab",
        "sql", "bash", "shell", "perl", "lua", "haskell", "elixir",
    ],
    "frameworks": [
        "react", "angular", "vue", "svelte", "django", "flask", "fastapi",
        "spring", "spring boot", ".net", "node.js", "express", "next.js",
        "rails", "laravel",
    ],
    "ml_and_data": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "spark", "airflow", "dbt", "mlflow", "kubeflow", "hugging face",
        "transformers", "langchain", "opencv",
    ],
    "devops_and_cloud": [
        "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "github actions", "gitlab ci", "aws", "gcp", "azure",
        "linux", "git", "ci/cd", "prometheus", "grafana",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "sqlite", "cassandra", "dynamodb", "bigquery", "snowflake",
        "neo4j", "kafka",
    ],
}
CASE_SENSITIVE_SKILLS: set[str] = {"R"}

# Flattened for matching
ALL_SKILLS: set[str] = set()
for _group in SKILLS.values():
    ALL_SKILLS.update(_group)

DOMAINS: list[str] = [
    "machine learning", "deep learning", "artificial intelligence",
    "nlp", "natural language processing", "computer vision",
    "data science", "data engineering", "mlops",
    "software engineering", "backend", "frontend", "full stack", "fullstack",
    "devops", "site reliability", "sre", "cloud computing",
    "cybersecurity", "information security",
    "fintech", "finance", "banking",
    "healthcare", "healthtech", "biotech", "bioinformatics",
    "e-commerce", "retail",
    "automotive", "aerospace", "energy",
    "telecommunications", "telecom",
    "education", "edtech",
    "legal", "legaltech",
    "marketing", "adtech",
    "supply chain", "logistics",
    "consulting",
    "robotics", "iot", "embedded systems",
    "blockchain", "web3",
    "gaming", "game development",
]

DOMAIN_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "sre": "site reliability",
}

CV_SECTION_HEADERS: set[str] = {
    # English
    "education", "experience", "work experience", "professional experience",
    "skills", "technical skills", "projects", "personal projects",
    "certifications", "languages", "interests", "references",
    "summary", "objective", "training", "awards", "publications",
    "volunteer", "hobbies", "contact", "achievements", "activities",
    # French
    "formation", "expérience", "expérience professionnelle",
    "compétences", "compétences techniques", "projets", "langues",
    "centres d'intérêt", "références", "diplômes",
}

# Seniority markers — checked in order, first match wins
SENIORITY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:junior|jr\.?|entry[\s-]?level|débutant|jeune diplômé|intern|stagiaire)\b", "junior"),
    (r"\b(?:mid[\s-]?level|intermédiaire)\b", "mid"),
    (r"\b(?:senior|sr\.?|expérimenté)\b", "senior"),
    (r"\b(?:lead|team lead|tech lead|chef d'équipe)\b", "lead"),
    (r"\b(?:principal|staff|distinguished)\b", "principal"),
    (r"\b(?:head of|director|directeur|vp |vice president|cto|cio)\b", "principal"),
]

# Years-of-experience fallback for seniority
YEARS_PATTERN = re.compile(
    r"(\d{1,2})\s*(?:\+\s*)?(?:years?|ans?|années?)\s*(?:of\s+)?(?:experience|expérience|d'expérience)?",
    re.IGNORECASE,
)

# Spoken languages — EN name and FR name both map to the same canonical label
SPOKEN_LANGUAGES: dict[str, str] = {
    # English names
    "english": "English", "french": "French", "german": "German",
    "spanish": "Spanish", "italian": "Italian", "portuguese": "Portuguese",
    "dutch": "Dutch", "arabic": "Arabic", "chinese": "Chinese",
    "mandarin": "Chinese", "japanese": "Japanese", "korean": "Korean",
    "russian": "Russian", "hindi": "Hindi", "turkish": "Turkish",
    "polish": "Polish", "swedish": "Swedish", "norwegian": "Norwegian",
    "danish": "Danish", "finnish": "Finnish", "greek": "Greek",
    "czech": "Czech", "hungarian": "Hungarian", "romanian": "Romanian",
    # French names
    "anglais": "English", "français": "French", "allemand": "German",
    "espagnol": "Spanish", "italien": "Italian", "portugais": "Portuguese",
    "néerlandais": "Dutch", "arabe": "Arabic", "chinois": "Chinese",
    "japonais": "Japanese", "coréen": "Korean", "russe": "Russian",
    "turc": "Turkish", "polonais": "Polish", "suédois": "Swedish",
    "norvégien": "Norwegian", "danois": "Danish", "finnois": "Finnish",
    "grec": "Greek", "tchèque": "Czech", "hongrois": "Hungarian",
    "roumain": "Romanian",
}


# ---------------------------------------------------------------------------
# Rule-based extractor
# ---------------------------------------------------------------------------

class RuleBasedExtractor(Extractor):
    """Extract profile facets via keyword matching and regex heuristics."""

    def extract(self, raw_text: str) -> dict[str, Any]:
        text_lower = raw_text.lower()
        return {
            "skills": self._extract_skills(text_lower, raw_text),
            "domains": self._extract_domains(text_lower),
            "seniority": self._extract_seniority(text_lower),
            "languages": self._extract_languages(text_lower),
        }

    def _extract_skills(self, text_lower: str, text_original: str) -> list[str]:
        found = []
        for skill in ALL_SKILLS:
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, text_lower):
                found.append(skill)
        for skill in CASE_SENSITIVE_SKILLS:
                    if skill == "R":
                        # Exclude R&D, R&D-like patterns
                        if re.search(r"\bR\b(?!\s*&)", text_original):
                            found.append(skill.lower())
                    else:
                        pattern = rf"\b{re.escape(skill)}\b"
                        if re.search(pattern, text_original):
                            found.append(skill.lower())
        return sorted(found)

    def _is_section_header(self, line: str) -> bool:
        """Check if a line looks like a CV section header."""
        stripped = line.strip().lower().rstrip(":")
        return stripped in CV_SECTION_HEADERS

    def _extract_domains(self, text: str) -> list[str]:
        # Strip section headers to avoid false positives
        lines = text.split("\n")
        body_text = "\n".join(
            line for line in lines if not self._is_section_header(line)
        )

        found = set()
        for domain in DOMAINS:
            pattern = rf"\b{re.escape(domain)}\b"
            if re.search(pattern, body_text):
                found.add(domain)

        # Check aliases — map to canonical name
        for alias, canonical in DOMAIN_ALIASES.items():
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, body_text):
                found.add(canonical)

        return sorted(found)

    def _extract_seniority(self, text: str) -> str:
        # Check explicit markers first
        for pattern, level in SENIORITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return level

        # Fallback: years of experience
        matches = YEARS_PATTERN.findall(text)
        if matches:
            max_years = max(int(y) for y in matches)
            if max_years < 1:
                return "junior"
            elif max_years < 3:
                return "junior"
            elif max_years < 6:
                return "mid"
            elif max_years < 10:
                return "senior"
            else:
                return "principal"

        return "junior"  # safe default

    def _extract_languages(self, text: str) -> list[str]:
        found = set()
        for keyword, canonical in SPOKEN_LANGUAGES.items():
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, text):
                found.add(canonical)
        return sorted(found)