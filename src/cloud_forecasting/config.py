"""Configuration loading and project-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML settings and resolve configured paths from the project root."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    config["_project_root"] = project_root
    config.setdefault("data", {})
    config.setdefault("outputs", {})
    for section, keys in {
        "data": ("readings", "metadata"),
        "outputs": ("root_predictions", "directory", "figures", "report"),
    }.items():
        for key in keys:
            value = config[section].get(key)
            if value is not None:
                candidate = Path(value).expanduser()
                config[section][key] = candidate if candidate.is_absolute() else project_root / candidate
    return config