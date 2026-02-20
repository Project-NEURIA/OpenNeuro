from __future__ import annotations

from pathlib import Path

BASE_DIR = Path.home() / "Documents" / "OpenNeuro"
PROJECTS_DIR = BASE_DIR / "projects"
CONFIG_PATH = BASE_DIR / "config.json"


def project_dir(name: str) -> Path:
    return PROJECTS_DIR / name


def graph_path(name: str) -> Path:
    return project_dir(name) / "graph.json"


def thumbnail_path(name: str) -> Path:
    return project_dir(name) / "thumbnail.png"
