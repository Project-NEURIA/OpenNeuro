from __future__ import annotations

import json
import shutil
import struct
import zlib
from pathlib import Path

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
                    has_thumbnail=(
                        PROJECTS_DIR / "graph.json" / "thumbnail.png"
                    ).exists(),
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
