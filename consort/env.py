"""Secrets from a local `.env`, for tools that read the environment (the Kaggle CLI, RunPod).

Only `KEY=value` lines are read; a variable already set in the environment wins, so CI and a
reader's shell override the file. Values are never logged.
"""

from __future__ import annotations

import os
from pathlib import Path


def load(path: Path = Path(".env")) -> list[str]:
    """Set the file's variables that are not already set; return the names it set."""
    if not path.exists():
        return []
    set_names = []
    for raw in path.read_text(encoding="utf8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            set_names.append(key)
    return set_names
