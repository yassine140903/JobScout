"""HTML cleaning and post-extraction text quality checks.

Two storage-layer defences:

``clean_description``  turns adapter-supplied markup into structured plain text
                      before it is written to ``jobs.description``.
``check_text_quality`` flags text that came out of an extractor mangled -
                      word-gluing, runaway tokens - before it is stored.
"""

from __future__ import annotations

import html as html_module
import logging
import re
from dataclasses import dataclass, field

from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

# Tags whose boundary becomes a single newline - items within one block.
# Table cells are included: without them "<td>a</td><td>b</td>" collapses to
# "ab", the same word-gluing class of bug this module exists to prevent.
LINE_BREAK_TAGS: frozenset[str] = frozenset({
    "br", "li", "tr", "td", "th", "dt", "dd", "option",
})

# Tags whose boundary becomes a blank line - separate blocks of prose. Keeping
# paragraphs visually separated is what lets a truncated description still show
# where the requirements section starts.
PARAGRAPH_TAGS: frozenset[str] = frozenset({
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "dl", "table", "thead", "tbody", "tfoot",
    "section", "article", "header", "footer", "aside", "nav",
    "blockquote", "pre", "hr", "form", "fieldset", "figure", "figcaption",
})

BLOCK_TAGS: frozenset[str] = LINE_BREAK_TAGS | PARAGRAPH_TAGS

# Removed with their contents, not unwrapped.
DROP_TAGS: tuple[str, ...] = ("script", "style")

_TEXT_NODE = "-text"
_MAX_UNESCAPE_PASSES = 5

_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_HORIZONTAL_WS_RE = re.compile(r"[ \t   ]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_NEWLINE_PAD_RE = re.compile(r"[ \t]*\n[ \t]*")


def _unescape_to_fixpoint(text: str) -> str:
    """Unescape entities repeatedly until stable.

    A single pass is not idempotent: ``clean(clean("&amp;amp;"))`` would differ
    from ``clean("&amp;amp;")``. Running to a fixpoint makes the output a
    fixpoint of unescape too, so ``clean`` is genuinely idempotent. It also
    repairs the double-escaped ``&amp;`` that several stored descriptions
    already contain.
    """
    for _ in range(_MAX_UNESCAPE_PASSES):
        unescaped = html_module.unescape(text)
        if unescaped == text:
            return text
        text = unescaped
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse horizontal runs, trim per line, cap blank runs at one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    text = _NEWLINE_PAD_RE.sub("\n", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_description(html: str | None) -> str:
    """Convert job-description markup to structured plain text.

    Preserves block structure as newlines (descriptions get truncated before
    they reach an LLM, and structure is what lets the requirements section
    survive the cut), drops ``<script>``/``<style>`` subtrees with their
    contents, unescapes entities, and normalizes whitespace.

    Idempotent, and a no-op on text that contains no markup.
    """
    if not html:
        return ""

    # Plain text: nothing to parse. Entities are still resolved so that
    # "Python &amp; SQL" reads correctly whether or not tags were present.
    if "<" not in html:
        return _normalize_whitespace(_unescape_to_fixpoint(html))

    tree = HTMLParser(html)
    if tree.body is None:
        return _normalize_whitespace(_unescape_to_fixpoint(html))

    # Replace rather than decompose: dropping the subtree outright would glue
    # the text on either side together ("Keep<script>..</script>this" -> "Keepthis").
    for tag in DROP_TAGS:
        for node in tree.css(tag):
            node.replace_with("\n")

    parts: list[str] = []
    for node in tree.body.traverse(include_text=True):
        if node.tag == _TEXT_NODE:
            parts.append(node.text_content or "")
        elif node.tag in PARAGRAPH_TAGS:
            parts.append("\n\n")
        elif node.tag in LINE_BREAK_TAGS:
            parts.append("\n")

    return _normalize_whitespace(_unescape_to_fixpoint("".join(parts)))


# ---------------------------------------------------------------------------
# Text quality guard
# ---------------------------------------------------------------------------

# A PDF extractor that merges words produces long tokens, a high mean token
# length, and tokens that fuse words with numbers ("across20containerizedservices").
#
# Caveat: at 25 this warns on ~2.8% of the current corpus, nearly all of it
# legitimate - German/Dutch/Swedish compounds ("Berufsunfähigkeitsversicherung")
# and CJK text, which has no word spaces at all. The checks only warn, so the
# noise is survivable; raise the threshold per-call if a corpus is compound-heavy.
MAX_TOKEN_LENGTH = 25
MAX_MEAN_TOKEN_LENGTH = 12.0
MAX_ALNUM_MIXED_RATIO = 0.15
MIN_TOKENS_FOR_RATIOS = 20  # below this, ratios are noise

_TOKEN_SPLIT_RE = re.compile(r"\s+")
_TOKEN_TRIM_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
# Matched anywhere in the token, not just at the start: markdown links arrive
# as "commitment](https://...)" with the URL embedded mid-token.
_URLISH_RE = re.compile(
    r"""(?:
        (?:https?|ftp)://
      | www\.
      | mailto:
      | [^\s@]+@[^\s@]+\.[^\s@]+          # email
      | [\w.-]+\.(?:com|org|net|io|dev|edu|gov|eu|fr|de|uk|co|ai|me)(?:/|$)
    )""",
    re.IGNORECASE | re.VERBOSE,
)
# Splits a token into its unbroken alphanumeric runs.
_ALNUM_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_HAS_DIGIT_RE = re.compile(r"\d")


@dataclass
class TextQualityReport:
    """Outcome of a post-extraction sanity check."""

    ok: bool
    n_tokens: int
    mean_token_length: float
    alnum_mixed_ratio: float
    long_tokens: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"tokens={self.n_tokens} mean_len={self.mean_token_length:.1f} "
            f"mixed_ratio={self.alnum_mixed_ratio:.2f} "
            f"long_tokens={len(self.long_tokens)}"
        )


def _is_urlish(token: str) -> bool:
    return bool(_URLISH_RE.search(token))


def longest_alnum_run(token: str) -> int:
    """Length of the longest run of letters/digits unbroken by punctuation.

    Measured instead of raw token length because legitimate long tokens are
    long thanks to separators - "Snowflake/Databricks/BigQuery",
    "informatiques/technologiques" - whereas extractor word-gluing produces
    one continuous run ("across20containerizedservices"). Raw length flagged
    7.7% of this corpus; this rule flags the gluing without the false alarms.
    """
    runs = _ALNUM_RUN_RE.findall(token)
    return max((len(r) for r in runs), default=0)


def tokenize(text: str) -> list[str]:
    """Whitespace tokens with leading/trailing punctuation trimmed."""
    tokens = []
    for raw in _TOKEN_SPLIT_RE.split(text):
        token = _TOKEN_TRIM_RE.sub("", raw)
        if token:
            tokens.append(token)
    return tokens


def check_text_quality(
    text: str,
    *,
    max_token_length: int = MAX_TOKEN_LENGTH,
    max_mean_token_length: float = MAX_MEAN_TOKEN_LENGTH,
    max_alnum_mixed_ratio: float = MAX_ALNUM_MIXED_RATIO,
) -> TextQualityReport:
    """Flag text that an extractor likely mangled.

    Suspicious when a token has an unbroken alphanumeric run longer than
    ``max_token_length`` without being a URL or email, when the mean token
    length is implausibly high, or when too many tokens fuse letters with digits.
    """
    tokens = tokenize(text)
    if not tokens:
        return TextQualityReport(
            ok=False,
            n_tokens=0,
            mean_token_length=0.0,
            alnum_mixed_ratio=0.0,
            reasons=["no tokens - text is empty or whitespace only"],
        )

    candidates = [t for t in tokens if not _is_urlish(t)]
    if not candidates:
        candidates = tokens

    long_tokens = [t for t in candidates if longest_alnum_run(t) > max_token_length]
    mean_length = sum(len(t) for t in candidates) / len(candidates)
    mixed = sum(
        1
        for t in candidates
        if _HAS_LETTER_RE.search(t) and _HAS_DIGIT_RE.search(t)
    )
    mixed_ratio = mixed / len(candidates)

    reasons: list[str] = []
    if long_tokens:
        reasons.append(
            f"{len(long_tokens)} token(s) with an unbroken run over "
            f"{max_token_length} chars"
        )
    # Ratio-based checks need enough tokens to mean anything.
    if len(candidates) >= MIN_TOKENS_FOR_RATIOS:
        if mean_length > max_mean_token_length:
            reasons.append(
                f"mean token length {mean_length:.1f} > {max_mean_token_length}"
            )
        if mixed_ratio > max_alnum_mixed_ratio:
            reasons.append(
                f"letter+digit token ratio {mixed_ratio:.2f} > {max_alnum_mixed_ratio}"
            )

    return TextQualityReport(
        ok=not reasons,
        n_tokens=len(candidates),
        mean_token_length=mean_length,
        alnum_mixed_ratio=mixed_ratio,
        long_tokens=long_tokens,
        reasons=reasons,
    )


def log_quality_warning(report: TextQualityReport, source: str) -> None:
    """Emit a warning naming the source and the offending tokens."""
    offenders = ", ".join(repr(t) for t in report.long_tokens[:5])
    if len(report.long_tokens) > 5:
        offenders += f", ... (+{len(report.long_tokens) - 5} more)"
    logger.warning(
        "Suspicious extracted text from %s: %s | %s%s",
        source,
        "; ".join(report.reasons),
        report.summary(),
        f" | offending tokens: {offenders}" if offenders else "",
    )
