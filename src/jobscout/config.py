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
        "weights": {
            "skills": 0.35,
            "domain": 0.30,
            "seniority": 0.20,
            "location": 0.15,
        },
    },
    "sources": [],
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