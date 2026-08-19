"""CV text extraction and profile enrichment."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from jobscout.textclean import (
    TextQualityReport,
    check_text_quality,
    log_quality_warning,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

# pdfplumber merges adjacent words when x_tolerance is too loose. The previous
# value of 2 glued this project's CV into 259 tokens averaging 17.1 chars
# ("PostgreSQLasdurablestore"); 1.5 yields 668 tokens averaging 6.7.
DEFAULT_PDF_X_TOLERANCE = 1.5

# Tried in order when the primary tolerance produces text that fails the
# quality guard. layout=True preserves column alignment at the cost of padding
# whitespace, so it is a last resort rather than the default.
PDF_FALLBACK_STRATEGIES: tuple[dict[str, Any], ...] = (
    {"x_tolerance": 1.0},
    {"x_tolerance": 1.5, "layout": True},
)


def extract_text(path: Path, *, x_tolerance: float | None = None) -> str:
    """Extract raw text from a PDF or DOCX file.

    Output is checked by the text-quality guard; a PDF that fails is retried
    with alternate extraction settings before the text is handed back.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, x_tolerance=x_tolerance)
    elif suffix in (".docx", ".doc"):
        text = _extract_docx(path)
        report = check_text_quality(text)
        if not report.ok:
            log_quality_warning(report, str(path))
        return text
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _pdf_page_text(path: Path, **kwargs: Any) -> str:
    """Run one pdfplumber extraction pass over every page."""
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(**kwargs)
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _quality_rank(report: TextQualityReport) -> tuple[int, float]:
    """Sort key for picking the least-mangled extraction. Lower is better."""
    return (len(report.long_tokens), report.mean_token_length)


def _extract_pdf(path: Path, x_tolerance: float | None = None) -> str:
    """Extract PDF text, falling back to alternate settings on poor quality."""
    primary = DEFAULT_PDF_X_TOLERANCE if x_tolerance is None else x_tolerance

    attempts: list[dict[str, Any]] = [{"x_tolerance": primary}]
    attempts += [s for s in PDF_FALLBACK_STRATEGIES if s != attempts[0]]

    best: tuple[tuple[int, float], str, TextQualityReport, dict] | None = None

    for i, kwargs in enumerate(attempts):
        text = _pdf_page_text(path, **kwargs)
        report = check_text_quality(text)
        if report.ok:
            if i > 0:
                logger.info(
                    "PDF extraction of %s recovered with %s (%s)",
                    path.name, kwargs, report.summary(),
                )
            return text

        rank = _quality_rank(report)
        if best is None or rank < best[0]:
            best = (rank, text, report, kwargs)
        logger.debug(
            "PDF extraction of %s with %s failed quality check: %s",
            path.name, kwargs, "; ".join(report.reasons),
        )

    # Every strategy failed - hand back the least-bad one, loudly.
    _, text, report, kwargs = best
    logger.warning(
        "All %d PDF extraction strategies failed the quality check for %s; "
        "using the least-mangled result (%s)",
        len(attempts), path.name, kwargs,
    )
    log_quality_warning(report, str(path))
    return text


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
    # The abstraction layer. Everything above names a product; a posting that
    # asks for "mise en production de modèles" and "tests unitaires" is asking
    # for real skills that no product name covers. Against the hand-labelled
    # gold entries the vocabulary scored precision 0.97 and recall 0.58 - it
    # was not wrong about what it found, it just only looked for tools.
    "concepts_and_practices": [
        "machine learning", "deep learning", "mlops",
        "rest api", "microservices", "model serving", "model monitoring",
        "feature store", "data drift",
        "iac", "ci/cd", "cloud architecture", "distributed computing",
        "data lake", "lakehouse", "data mesh", "data warehouse",
        "etl", "elt", "data governance", "data quality",
        "unit testing", "tdd",
    ],
}
CASE_SENSITIVE_SKILLS: set[str] = {"R"}

# Surface forms that mean the same skill. The value is what gets reported, so
# a French posting and an English one produce the same skill string and the
# same embedding text. Only forms actually seen in the corpus languages.
SKILL_ALIASES: dict[str, str] = {
    # machine learning
    "apprentissage automatique": "machine learning",
    "apprentissage statistique": "machine learning",
    "maschinelles lernen": "machine learning",
    # deep learning
    "apprentissage profond": "deep learning",
    "réseaux de neurones": "deep learning",
    "neuronale netze": "deep learning",
    # infrastructure as code
    "infrastructure as code": "iac",
    "infrastructure-as-code": "iac",
    # rest. French inverts the noun phrase ("API REST"), and English writes
    # "RESTful" as often as "REST API". Separator and trailing-plural variation
    # is handled by skill_pattern, so only genuinely distinct word orders and
    # stems are listed here - "rest-api", "REST  API" and "rest apis" all fall
    # out of the generator.
    "api rest": "rest api",
    "apis rest": "rest api",
    "api restful": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "web api": "rest api",
    "api web": "rest api",
    "rest-schnittstelle": "rest api",
    # storage and modelling
    "entrepôt de données": "data warehouse",
    "datenlager": "data warehouse",
    "lac de données": "data lake",
    "datensee": "data lake",
    "maillage de données": "data mesh",
    # governance
    "gouvernance des données": "data governance",
    "datenverwaltung": "data governance",
    "qualité des données": "data quality",
    "datenqualität": "data quality",
    # testing
    "tests unitaires": "unit testing",
    "test unitaire": "unit testing",
    "unit tests": "unit testing",
    "komponententests": "unit testing",
    "test driven development": "tdd",
    "développement piloté par les tests": "tdd",
    # architecture and scale
    "architecture cloud": "cloud architecture",
    "cloud-architektur": "cloud architecture",
    "calcul distribué": "distributed computing",
    "calcul réparti": "distributed computing",
    "verteiltes rechnen": "distributed computing",
    "micro-services": "microservices",
    "mikroservices": "microservices",
    # operations
    "mise en production de modèles": "model serving",
    "industrialisation des modèles": "mlops",
    "surveillance des modèles": "model monitoring",
    "dérive des données": "data drift",
    "modelldrift": "data drift",
    "magasin de features": "feature store",
    "intégration continue": "ci/cd",
    "déploiement continu": "ci/cd",
    "kontinuierliche integration": "ci/cd",
}

# Flattened for matching
ALL_SKILLS: set[str] = set()
for _group in SKILLS.values():
    ALL_SKILLS.update(_group)


def skill_pattern(term: str) -> re.Pattern[str]:
    """Word-bounded, case-insensitive pattern for a vocabulary term.

    Three tolerances, each measured against the corpus before being added:

      * whitespace, hyphens or underscores between the words of a phrase, so
        "rest api", "REST-API" and "REST  API" all match one entry;
      * an optional English plural on the final token, which brings in
        "data lakes", "feature stores" and "rest apis" - 54 extra document
        matches across 8 of the 49 multi-word terms, every one of them a
        correct plural rather than a different word;
      * case, via IGNORECASE. These patterns used to be lowercase literals
        that silently matched nothing unless the caller had already lowered
        its input. That precondition was invisible at the call site and cost
        a wrong measurement; the flag removes it.
    """
    parts = [re.escape(part) for part in term.split()]
    body = r"[\s\-_]+".join(parts)
    return re.compile(r"\b" + body + r"(?:es|s)?\b", re.IGNORECASE)


# Compiled once: the extractor runs this vocabulary over every stored job.
SKILL_PATTERNS: dict[str, re.Pattern[str]] = {
    skill: skill_pattern(skill) for skill in ALL_SKILLS
}
SKILL_ALIAS_PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    alias: (skill_pattern(alias), canonical)
    for alias, canonical in SKILL_ALIASES.items()
}

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
        result = self.extract_from_text(raw_text)
        text_lower = raw_text.lower()
        result["languages"] = self._extract_languages(text_lower)
        return result

    def extract_from_text(self, raw_text: str) -> dict[str, Any]:
        """Extract skills/domains/seniority from any text (CV, job description, etc.)."""
        text_lower = raw_text.lower()
        return {
            "skills": self._extract_skills(text_lower, raw_text),
            "domains": self._extract_domains(text_lower),
            "seniority": self._extract_seniority(text_lower),
        }

    def _extract_skills(self, text_lower: str, text_original: str) -> list[str]:
        # The patterns are case-insensitive, so matching no longer depends on
        # this argument being lowered. The order of the two arguments still
        # does: the case-sensitive branch below reads text_original, and a
        # caller that swapped them would break "R" detection silently. Say so
        # rather than carrying on.
        if text_lower != text_lower.lower():
            logger.warning(
                "_extract_skills: text_lower argument is not lowercased "
                "(%d of %d chars differ) - the caller may have swapped the two "
                "arguments, which breaks case-sensitive skill detection",
                sum(1 for a, b in zip(text_lower, text_lower.lower()) if a != b),
                len(text_lower),
            )

        found = []
        for skill, pattern in SKILL_PATTERNS.items():
            if pattern.search(text_lower):
                found.append(skill)
        # A French posting saying "apprentissage automatique" and an English one
        # saying "machine learning" must produce the same skill string, or the
        # facet embeddings drift apart on wording rather than content.
        for pattern, canonical in SKILL_ALIAS_PATTERNS.values():
            if canonical not in found and pattern.search(text_lower):
                found.append(canonical)
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


# ---------------------------------------------------------------------------
# Experience duration from CV date ranges (M7b)
# ---------------------------------------------------------------------------
#
# Advisory only. The number this produces is a suggestion for the user to look
# at and confirm; it never becomes the scored candidate_years on its own. A CV
# writes dates for degrees, certifications and target start dates too, and no
# amount of pattern work reliably tells those apart from employment.

# Headers that open the section we are willing to count.
EXPERIENCE_HEADERS: set[str] = {
    "experience", "work experience", "professional experience",
    "employment", "employment history", "career",
    "expérience", "expériences", "expérience professionnelle",
    "expériences professionnelles", "parcours professionnel",
}

# Every other known header closes it. Education in particular must not be
# counted: a degree spanning 2022-2027 is not five years of employment.
_SECTION_STOP_HEADERS: set[str] = CV_SECTION_HEADERS - EXPERIENCE_HEADERS

MONTH_NAMES: dict[str, int] = {
    "jan": 1, "january": 1, "janv": 1, "janvier": 1,
    "feb": 2, "february": 2, "fev": 2, "fév": 2, "fevr": 2, "févr": 2,
    "fevrier": 2, "février": 2,
    "mar": 3, "march": 3, "mars": 3,
    "apr": 4, "april": 4, "avr": 4, "avril": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "juin": 6,
    "jul": 7, "july": 7, "juil": 7, "juillet": 7,
    "aug": 8, "august": 8, "aou": 8, "aoû": 8, "aout": 8, "août": 8,
    "sep": 9, "sept": 9, "september": 9, "septembre": 9,
    "oct": 10, "october": 10, "octobre": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "december": 12, "déc": 12, "decembre": 12, "décembre": 12,
}

_MONTH_ALT = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))
_PRESENT_ALT = (
    r"present|présent|now|today|current|currently|ongoing|"
    r"actuel|actuelle|aujourd'hui|en cours|à ce jour"
)
_SEP = r"\s*(?:[-–—]|to|until|à|au|jusqu'à|jusqu'au)\s*"
_YEAR = r"(?:19|20)\d{2}"

# "jan 2021 - mar 2023", "2021 - 2024", "2021 – present"
RANGE_PATTERN = re.compile(
    rf"(?:(?P<m1>{_MONTH_ALT})\.?\s+)?(?P<y1>{_YEAR})"
    rf"{_SEP}"
    rf"(?:(?P<present>{_PRESENT_ALT})|(?:(?P<m2>{_MONTH_ALT})\.?\s+)?(?P<y2>{_YEAR}))",
    re.IGNORECASE,
)

# "depuis 2022", "since jan 2022" — open-ended, runs to today.
SINCE_PATTERN = re.compile(
    rf"\b(?:depuis|since|from)\s+(?:(?P<m1>{_MONTH_ALT})\.?\s+)?(?P<y1>{_YEAR})"
    rf"(?!{_SEP}(?:{_MONTH_ALT}|{_YEAR}|{_PRESENT_ALT}))",
    re.IGNORECASE,
)

_YEAR_ANYWHERE = re.compile(_YEAR)


@dataclass
class ExperienceParse:
    """What the date parser found. `years` is a suggestion, not a verdict."""

    years: float
    ranges: list[str]        # the spans that were counted, as written
    ignored: list[str]       # date-bearing lines it could not turn into a span

    def summary(self) -> str:
        parts = [f"{self.years:.2f} years from {len(self.ranges)} date range(s)"]
        if self.ignored:
            parts.append(f"{len(self.ignored)} date-bearing line(s) not parsed")
        return "; ".join(parts)


def _month_index(year: int, month: int) -> int:
    """Absolute month number, so date arithmetic is plain integer subtraction."""
    return year * 12 + (month - 1)


def experience_section(raw_text: str) -> str:
    """Return only the professional-experience part of a CV.

    Falls back to an empty string when no experience header is present: better
    to suggest nothing than to sum every date on the page.
    """
    lines = raw_text.split("\n")
    kept: list[str] = []
    inside = False
    for line in lines:
        header = line.strip().lower().rstrip(":").strip()
        if header in EXPERIENCE_HEADERS:
            inside = True
            continue
        if inside and header in _SECTION_STOP_HEADERS:
            inside = False
            continue
        if inside:
            kept.append(line)
    return "\n".join(kept)


def _merge_spans(spans: list[tuple[int, int]]) -> int:
    """Total months covered by the union of spans — concurrent roles count once."""
    if not spans:
        return 0
    total = 0
    spans = sorted(spans)
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:            # overlapping or touching
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def parse_experience_years(
    raw_text: str, today: date | None = None,
) -> ExperienceParse:
    """Sum date ranges in the experience section, merging overlaps.

    Conventions, all deliberately conservative:
      * a year-only endpoint means January of that year, so "2021 - 2024"
        is three years rather than four;
      * a month-qualified end includes that whole month;
      * open-ended ranges ("present", "depuis 2022") run to today;
      * anything ending in the future is clamped to today.
    """
    today = today or date.today()
    today_idx = _month_index(today.year, today.month)

    section = experience_section(raw_text)
    if not section.strip():
        return ExperienceParse(0.0, [], [])

    spans: list[tuple[int, int]] = []
    counted: list[str] = []
    consumed: set[int] = set()   # line numbers that produced a span

    def line_of(pos: int) -> int:
        return section.count("\n", 0, pos)

    for match in RANGE_PATTERN.finditer(section):
        start = _month_index(
            int(match.group("y1")),
            MONTH_NAMES[match.group("m1").lower()] if match.group("m1") else 1,
        )
        if match.group("present"):
            end = today_idx + 1
        elif match.group("m2"):
            # An end month is inclusive: "mar 2023" means through March.
            end = _month_index(int(match.group("y2")),
                               MONTH_NAMES[match.group("m2").lower()]) + 1
        else:
            end = _month_index(int(match.group("y2")), 1)

        end = min(end, today_idx + 1)
        consumed.add(line_of(match.start()))
        if end > start:
            spans.append((start, end))
            counted.append(match.group(0).strip())

    for match in SINCE_PATTERN.finditer(section):
        if line_of(match.start()) in consumed:
            continue  # already counted as an explicit range on this line
        start = _month_index(
            int(match.group("y1")),
            MONTH_NAMES[match.group("m1").lower()] if match.group("m1") else 1,
        )
        consumed.add(line_of(match.start()))
        if today_idx + 1 > start:
            spans.append((start, today_idx + 1))
            counted.append(match.group(0).strip())

    # Surface what carried a date but yielded nothing, so silent drops are visible.
    ignored = [
        line.strip()
        for i, line in enumerate(section.split("\n"))
        if i not in consumed and _YEAR_ANYWHERE.search(line)
    ]

    months = _merge_spans(spans)
    return ExperienceParse(round(months / 12.0, 2), counted, ignored)


def resolve_candidate_years(
    profile: Any, config: dict | None = None,
) -> tuple[float | None, str]:
    """Decide the candidate's years of experience. Returns (years, source).

    Explicit settings win, in order: config, then the profile record. The
    parsed suggestion is never used here — it is shown to the user and waits
    to be confirmed. Unset returns None, which the scorer treats as neutral.
    """
    if config:
        configured = (config.get("profile") or {}).get("candidate_years")
        if configured is not None:
            return float(configured), "config"

    stored = None
    if profile is not None:
        try:
            stored = profile["candidate_years"]
        except (KeyError, IndexError, TypeError):
            stored = profile.get("candidate_years") if hasattr(profile, "get") else None
    if stored is not None:
        return float(stored), "profile"

    return None, "unset"

# ---------------------------------------------------------------------------
# Years requirement from a job description (M7c)
# ---------------------------------------------------------------------------
#
# The structured field is unreliable: WTTJ returns experience_level_minimum
# null for postings whose prose states a requirement outright ("Vous disposez
# d'au moins 3 ans d'expérience..."). This reads the prose.
#
# The design is conservative on purpose. A number of years is only a
# requirement when the surrounding text says it is about experience, so every
# candidate match has to survive two context checks before it counts:
#
#   * an experience term must sit within PROXIMITY_WINDOW characters, which
#     rules out company age ("Simplon existe depuis 3 ans"), contract lengths
#     ("CDD de 2 ans") and programme durations ("formation sur 2 ans");
#   * if an education term sits closer to the number than the experience term
#     does, the number belongs to a degree, not a job, and is dropped.
#
# Nothing here guesses. When no pattern fires, or when the guards reject every
# candidate, the answer is None and the caller falls through to the next layer.

PROXIMITY_WINDOW = 60        # chars either side of a match, for the guards
SNIPPET_WINDOW = 40          # chars either side, for the snippet shown in the UI
MAX_PLAUSIBLE_YEARS = 20.0   # above this it is an age or a company anniversary
#                            ("backed by over 25 years of experience" clears the
#                            proximity guard on its own, so the cap has to catch it)

# Spelled out only as far as ten. Beyond that the words are rare in postings
# and the false-friend risk climbs faster than the recall gain.
NUMBER_WORDS: dict[str, dict[str, float]] = {
    "en": {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    },
    "fr": {
        "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
        "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    },
    "de": {
        "ein": 1, "eine": 1, "einem": 1, "einer": 1, "zwei": 2, "drei": 3,
        "vier": 4, "fünf": 5, "fuenf": 5, "sechs": 6, "sieben": 7,
        "acht": 8, "neun": 9, "zehn": 10,
    },
}

# The noun that turns a bare number into a duration, per language.
YEAR_UNITS: dict[str, str] = {
    "en": r"years?",
    "fr": r"ans?|ann[ée]es?",
    "de": r"jahre?n?",
}

# "3 à 5 ans", "3-5 years", "2/4 ans", "3 bis 5 Jahre" — a range states its floor first.
_RANGE_SEP = r"[-–—/]|à|a|to|bis|and|et|ou|or"

# Markdown survives into stored descriptions ("**3 ans d'expérience**"), so
# emphasis characters are allowed to sit between the number and the unit.
_EMPHASIS = r"[\s*_]*"


def _years_pattern(lang: str) -> re.Pattern[str]:
    """Number (or number word), optional range or '+', then that language's unit."""
    words = "|".join(sorted(NUMBER_WORDS[lang], key=len, reverse=True))
    # Digit runs are bounded on both sides so "2024" cannot yield a "20".
    number = rf"(?:(?<!\d)\d{{1,2}}(?:[.,]\d)?(?!\d)|\b(?:{words})\b)"
    return re.compile(
        rf"(?P<n1>{number})"
        rf"(?:\s*(?:{_RANGE_SEP})\s*(?P<n2>{number}))?"
        # The "+" is a floor marker and may follow either end of a range, so
        # "3 a 7+ ans" still reports its floor of 3 rather than restarting at 7.
        rf"\s*\+?"
        rf"{_EMPHASIS}"
        rf"(?:{YEAR_UNITS[lang]})\b",
        re.IGNORECASE,
    )


YEARS_PATTERNS: dict[str, re.Pattern[str]] = {
    lang: _years_pattern(lang) for lang in YEAR_UNITS
}

# What has to be nearby for a number of years to be about a job.
EXPERIENCE_TERM_RE = re.compile(
    r"exp[ée]rien|exp[ée]riment|erfahrung|erfahren", re.IGNORECASE,
)

# What, if it is nearer than the experience term, means the years belong to a
# course of study instead.
EDUCATION_TERM_RE = re.compile(
    r"dipl[ôo]m|formation|cursus|bac\s*\+|degree|studium|studien", re.IGNORECASE,
)

# A sentence end, which the window is not allowed to cross. Without this the
# window reaches into the neighbouring sentence and borrows its vocabulary:
# "Simplon existe depuis 1 an. Vous justifiez de 4 ans d'expérience" would
# read the company's age as a requirement of one year. Requires trailing
# whitespace, so "1.5 ans" and "min." mid-line are not boundaries.
SENTENCE_END_RE = re.compile(r"[.!?…](?=\s)")


@dataclass
class YearsRequirement:
    """A years figure read out of a description, with the text that produced it."""

    years: float
    snippet: str      # ~80 chars of context, for eyeballing a bad match
    matched: str      # the span the pattern actually matched
    language: str     # which language's patterns fired

    def summary(self) -> str:
        return f"{self.years:g}y [{self.language}] from {self.matched!r}"


def _word_or_number(token: str | None, lang: str) -> float | None:
    """Turn a matched number token into years. None when it is not a number."""
    if not token:
        return None
    token = token.strip()
    if token[0].isdigit():
        try:
            return float(token.replace(",", "."))
        except ValueError:
            return None
    word = NUMBER_WORDS[lang].get(token.lower())
    return None if word is None else float(word)


def _distance(span: tuple[int, int], other: tuple[int, int]) -> int:
    """Gap in characters between two spans; 0 when they touch or overlap."""
    return max(0, span[0] - other[1], other[0] - span[1])


def _nearest(pattern: re.Pattern[str], window: str, span: tuple[int, int]) -> int | None:
    """Distance from `span` to the closest match of `pattern` inside `window`."""
    distances = [_distance(span, m.span()) for m in pattern.finditer(window)]
    return min(distances) if distances else None


def _sentence_window(text: str, start: int, end: int) -> tuple[int, int]:
    """The context a match is allowed to draw on: its window, clipped to its sentence."""
    lo = max(0, start - PROXIMITY_WINDOW)
    hi = min(len(text), end + PROXIMITY_WINDOW)

    boundaries = list(SENTENCE_END_RE.finditer(text, lo, start))
    if boundaries:
        lo = boundaries[-1].end()
    closing = SENTENCE_END_RE.search(text, end, hi)
    if closing:
        hi = closing.start()
    return lo, hi


def _context_supports(text: str, start: int, end: int) -> bool:
    """The proximity guard: is this number of years about work experience?

    True only when an experience term is within the window *and* no education
    term sits closer to the number than it does. Ties go to education, so
    "formation de 3 ans" loses even where "expérience" is equidistant. The
    window never reaches past a sentence end, so a number cannot borrow the
    vocabulary of the sentence beside it.
    """
    lo, hi = _sentence_window(text, start, end)
    window = text[lo:hi]
    span = (start - lo, end - lo)

    experience_at = _nearest(EXPERIENCE_TERM_RE, window, span)
    if experience_at is None:
        return False
    education_at = _nearest(EDUCATION_TERM_RE, window, span)
    return education_at is None or experience_at < education_at


def _snippet(text: str, start: int, end: int) -> str:
    """~80 characters of context around a match, on one line."""
    lo = max(0, start - SNIPPET_WINDOW)
    hi = min(len(text), end + SNIPPET_WINDOW)
    body = " ".join(text[lo:hi].split())
    prefix = "…" if lo else ""
    suffix = "…" if hi < len(text) else ""
    return f"{prefix}{body}{suffix}"


def _scan(text: str, lang: str) -> list[YearsRequirement]:
    """Every supported requirement this language's patterns find in the text."""
    found: list[YearsRequirement] = []
    for match in YEARS_PATTERNS[lang].finditer(text):
        values = [
            v for v in (
                _word_or_number(match.group("n1"), lang),
                _word_or_number(match.group("n2"), lang),
            )
            if v is not None
        ]
        if not values:
            continue
        years = min(values)                       # a range states its floor
        if not 0.0 <= years <= MAX_PLAUSIBLE_YEARS:
            continue
        if not _context_supports(text, match.start(), match.end()):
            continue
        found.append(YearsRequirement(
            years=years,
            snippet=_snippet(text, match.start(), match.end()),
            matched=" ".join(match.group(0).split()),
            language=lang,
        ))
    return found


def find_required_years(
    text: str | None, lang_hint: str | None = None,
) -> YearsRequirement | None:
    """Lowest stated experience requirement in a job description, with evidence.

    `lang_hint` is the posting's own language code. When it names a language we
    have patterns for and those patterns match, the other languages are not
    consulted — that stops a stray English boilerplate line ("2 years of
    experience with our tooling") from lowering the floor a French posting
    stated in French. Without a usable hint, every language is scanned and the
    lowest figure wins.

    Returns None whenever nothing survives the guards. Never guesses.
    """
    if not text:
        return None

    hinted = (lang_hint or "").strip().lower()[:2]
    stages: list[list[str]] = []
    if hinted in YEARS_PATTERNS:
        stages.append([hinted])
    stages.append(list(YEARS_PATTERNS))

    for langs in stages:
        candidates = [req for lang in langs for req in _scan(text, lang)]
        if candidates:
            # The posting is stating a floor; the lowest match is that floor.
            return min(candidates, key=lambda r: r.years)
    return None


def parse_required_years(text: str, lang_hint: str | None = None) -> float | None:
    """Years of experience a description asks for, or None when it asks for none.

    The number half of :func:`find_required_years`, which also returns the
    snippet that produced it.
    """
    found = find_required_years(text, lang_hint)
    return None if found is None else found.years
