"""Load and validate config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

DEFAULTS: dict[str, Any] = {
    "db_path": "jobscout.db",
    "model": {
        "name": "intfloat/multilingual-e5-base",
    },
    "extraction": {
        # pdfplumber word-merge tolerance. Too loose and adjacent words fuse
        # ("PostgreSQLasdurablestore"); tune per CV font without a code change.
        "pdf_x_tolerance": 1.5,
    },
    "profile": {
        # Years of professional experience, used by the seniority comparison.
        # Authoritative: CV date parsing only ever offers a suggestion, which
        # is reported but never written here. Left as None so a fresh install
        # does not silently claim zero experience — unset means the seniority
        # facet stays neutral and filters nothing.
        "candidate_years": None,
    },
    "scoring": {
        # Must stay in sync with matching.DEFAULT_WEIGHTS. Seniority is not a
        # weight — it is a multiplier applied on top of the weighted base score.
        "weights": {
            "skills": 0.60,
            "domain": 0.40,
        },
        "seniority": {
            # Years short of a requirement before a job is filtered out rather
            # than merely penalised. Overqualification never filters.
            "gate_years": 2.0,
            # Whether a title-inferred requirement may filter. Off: a guess
            # should cost rank position, not remove the job from view.
            "filter_on_inferred": False,
        },
    },
    "sources": [
        {
            "name": "wttj",
            "adapter": "wttj",
            "enabled": True,
            "keywords": [],
            "max_pages": 5,
            # The search index carries no job descriptions — only a short
            # requirements blurb. The real text needs one request per posting,
            # so this is the kill switch if that endpoint starts refusing
            # traffic, and the cap on how hard we hit it.
            "fetch_details": True,
            "detail_concurrency": 8,
        },
        {
            "name": "eures",
            "adapter": "eures",
            "enabled": True,
            "keywords": [],
            "locations": [],
            "max_pages": 3,
            "fetch_details": False,
            # Postings in other languages are fetched and discarded: the skill
            # vocabulary has no coverage for them, so they score on an empty
            # skill set. null or [] keeps every language.
            "languages": ["fr", "en", "de"],
        },
        {
            "name": "euraxess",
            "adapter": "euraxess",
            "enabled": False,
            "keyword": None,
            "countries": [],
            "research_fields": [],
            "max_pages": 3,
            "delay": 1.5,
        },
    ],
}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load config from YAML file, falling back to defaults for missing keys."""
    config = dict(DEFAULTS)

    if path.exists():
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged