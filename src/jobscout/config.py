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
    "scoring": {
        # Must stay in sync with matching.DEFAULT_WEIGHTS. Seniority is not a
        # weight — it is a multiplier applied on top of the weighted base score.
        "weights": {
            "skills": 0.60,
            "domain": 0.40,
        },
    },
    "sources": [
        {
            "name": "wttj",
            "adapter": "wttj",
            "enabled": True,
            "keywords": [],
            "max_pages": 5,
        },
        {
            "name": "eures",
            "adapter": "eures",
            "enabled": True,
            "keywords": [],
            "locations": [],
            "max_pages": 3,
            "fetch_details": False,
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