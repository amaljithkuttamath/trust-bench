"""Experiment config loading and validation."""

import csv
import json
from pathlib import Path

import yaml

from trust_bench.models.base import ConfigError

_REQUIRED_KEYS = ("probe", "model")


def load_config(path: str) -> dict:
    """Load and validate top-level experiment config structure.

    Supports two prompt formats:
    1. Inline prompts in YAML (prompts: [...])
    2. Eval set reference (eval_set: "path/to/data.csv" or "path/to/data.jsonl")
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(config).__name__}")
    for key in _REQUIRED_KEYS:
        if key not in config:
            raise ConfigError(f"Config missing required key: '{key}'")

    if "eval_set" in config:
        csv_path = Path(path).parent / config["eval_set"]
        config["prompts"] = load_eval_set(str(csv_path))

    return config


def load_eval_set(path: str) -> list[dict]:
    """Load an eval set (CSV or JSONL) into a list of row dicts."""
    path_obj = Path(path)
    if path_obj.suffix == ".jsonl":
        return _load_jsonl(path)
    elif path_obj.suffix == ".csv":
        return _load_csv(path)
    else:
        raise ConfigError(f"Unsupported eval set format: {path_obj.suffix}. Use .csv or .jsonl")


def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ConfigError(f"Eval set is empty: {path}")
    return rows


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ConfigError(f"Eval set is empty: {path}")
    return rows
