from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

from src.core.config import PROJECTS_DIR, CONFIG_PATH, AppConfig
from src.core.graph import Graph
from src.api.project.dto import ProjectSummary
from src.core.graph import GraphManager


def _write_placeholder_png(path: str | Path) -> None:
    """Write a minimal 1x1 transparent PNG."""
    raw = b"\x00\x00\x00\x00\x00"  # filter byte + 1 RGBA pixel (transparent)
    compressed = zlib.compress(raw)

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr_data)
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def list_projects() -> list[ProjectSummary]:
    if not PROJECTS_DIR.exists():
        return []
    results: list[ProjectSummary] = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        if d.is_dir():
            results.append(
                ProjectSummary(
                    name=d.name,
                    has_thumbnail=(d / "thumbnail.png").exists(),
                )
            )
    return results


def create_project(name: str) -> None:
    d = PROJECTS_DIR / name
    gp = d / "graph.json"
    tp = d / "thumbnail.png"

    d.mkdir(parents=True, exist_ok=True)
    if not gp.exists():
        gp.write_text('{"nodes": {}, "edges": []}')
    if not tp.exists():
        _write_placeholder_png(tp)


def delete_project(name: str) -> None:
    d = PROJECTS_DIR / name
    if d.exists():
        shutil.rmtree(d)


def start_project(name: str, manager: GraphManager) -> None:
    path = PROJECTS_DIR / name / "graph.json"
    if not path.exists():
        raise ValueError("error starting project")
    data = json.loads(path.read_text())
    graph = Graph.model_validate(data)
    manager.reset(graph)
    CONFIG_PATH.write_text(AppConfig(current_project=name).model_dump_json(indent=2))


def close_project(manager: GraphManager) -> None:
    manager.reset(Graph(edges=[], nodes={}))
    CONFIG_PATH.write_text(AppConfig(current_project=None).model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

_PATH_SUFFIXES = {
    ".pt",
    ".pth",
    ".pkl",
    ".bin",
    ".onnx",
    ".safetensors",
    ".png",
    ".jpg",
    ".jpeg",
    ".wav",
    ".mp3",
    ".json",
    ".txt",
    ".yaml",
    ".yml",
    ".vmd",
}


def _looks_like_path(value: str) -> bool:
    """Heuristic: a string looks like a file path if it contains a separator
    and points to an existing file, or has a known model/asset extension."""
    if not isinstance(value, str) or not value:
        return False
    p = Path(value)
    # Absolute path that exists
    if p.is_absolute() and p.is_file():
        return True
    # Relative path with a known extension (e.g. "assets/dart_control/foo.pt")
    if os.sep in value or "/" in value:
        if p.suffix.lower() in _PATH_SUFFIXES:
            return True
    return False


def _collect_paths(
    obj: Any, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], str]]:
    """Walk a JSON-like structure and return (key_path, value) for path-like strings."""
    results: list[tuple[tuple[str, ...], str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(_collect_paths(v, prefix + (k,)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            results.extend(_collect_paths(v, prefix + (str(i),)))
    elif isinstance(obj, str) and _looks_like_path(obj):
        results.append((prefix, obj))
    return results


def _set_nested(obj: Any, key_path: tuple[str, ...], value: str) -> None:
    """Set a value in a nested dict/list given a key path."""
    for part in key_path[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    last = key_path[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value


def export_project(name: str) -> tuple[str, list[str]]:
    """Prepare a project for sharing by copying assets and rewriting paths.

    Returns (project_dir, list_of_copied_asset_filenames).
    """
    project_dir = PROJECTS_DIR / name
    graph_path = project_dir / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Project '{name}' not found")

    data = json.loads(graph_path.read_text())
    assets_dir = project_dir / "assets"
    copied: list[str] = []

    for _node_id, node in data.get("nodes", {}).items():
        init_args = node.get("init_args", {})
        paths = _collect_paths(init_args)
        for key_path, raw_path in paths:
            src = Path(raw_path)
            # Try resolving relative to backend working dir if not absolute
            if not src.is_absolute():
                # Could be relative to the backend repo root
                candidates = [
                    Path.cwd() / raw_path,
                    Path(__file__).resolve().parent.parent.parent.parent / raw_path,
                ]
                src = next((c for c in candidates if c.is_file()), src)

            if not src.is_file():
                continue

            # Determine a unique filename in assets/
            dest_name = src.name
            dest = assets_dir / dest_name
            # Avoid collisions by prefixing with parent dir name
            if dest.exists() and not dest.samefile(src):
                dest_name = f"{src.parent.name}_{src.name}"
                dest = assets_dir / dest_name

            # Already in the project assets dir — skip copy
            if dest.exists() and dest.samefile(src):
                _set_nested(init_args, key_path, f"assets/{dest_name}")
                continue

            assets_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(dest_name)

            # Rewrite the path in init_args to relative
            _set_nested(init_args, key_path, f"assets/{dest_name}")

    # Save the rewritten graph
    graph_path.write_text(json.dumps(data, indent=2))

    return str(project_dir), copied


def import_project(git_url: str) -> str:
    """Clone a git repository into the projects directory.

    Returns the project name (repo directory name).
    """
    # Derive project name from the git URL
    repo_name = git_url.rstrip("/").rsplit("/", 1)[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    dest = PROJECTS_DIR / repo_name
    if dest.exists():
        raise FileExistsError(f"Project '{repo_name}' already exists")

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", "--depth", "1", git_url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )

    # Verify it has a graph.json
    if not (dest / "graph.json").exists():
        shutil.rmtree(dest)
        raise FileNotFoundError(
            "Cloned repository does not contain a graph.json at root"
        )

    # Write a placeholder thumbnail if none exists
    if not (dest / "thumbnail.png").exists():
        _write_placeholder_png(dest / "thumbnail.png")

    return repo_name
