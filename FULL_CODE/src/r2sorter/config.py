"""Configuration loading: one YAML file plus two optional environment overrides."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=8)
def _read_yaml(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return loaded


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the pipeline configuration (a fresh copy, safe to modify)."""
    chosen = path or os.environ.get("R2_CONFIG") or DEFAULT_CONFIG
    cfg = copy.deepcopy(_read_yaml(str(Path(chosen).resolve())))
    data_override = os.environ.get("R2_DATA_DIR")
    if data_override:
        cfg["paths"]["data_dir"] = data_override
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Turn a path entry from the config into an absolute path (relative paths start at FULL_CODE/)."""
    raw = Path(cfg["paths"][key])
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def data_file(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path of one of the data files named in the config."""
    return resolve_path(cfg, "data_dir") / cfg["paths"][key]
