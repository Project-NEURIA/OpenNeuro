from __future__ import annotations

import json
import shutil
import struct
import zlib
from pathlib import Path

from src.api.graph.domain.graph import Graph, Node, Edge
from src.api.graph.save.dto import SavedGraph
from src.api.project.config import AppConfig, save_config
from src.api.project.dto import ProjectSummary
from src.api.project.paths import PROJECTS_DIR, graph_path, project_dir, thumbnail_path


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
                    has_thumbnail=thumbnail_path(d.name).exists(),
                )
            )
    return results


def create_project(name: str) -> None:
    d = project_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    gp = graph_path(name)
    if not gp.exists():
        gp.write_text('{"nodes": {}, "edges": []}')
    tp = thumbnail_path(name)
    if not tp.exists():
        _write_placeholder_png(tp)


def delete_project(name: str) -> None:
    d = project_dir(name)
    if d.exists():
        shutil.rmtree(d)


def start_project(name: str, config: AppConfig, current_graph: Graph | None = None) -> Graph:
    if current_graph is not None:
        from src.api.graph.run.service import stop_all

        stop_all(current_graph)

    config.current_project = name
    save_config(config)
    path = graph_path(name)
    if not path.exists():
        return Graph(edges=[], nodes={})

    data = json.loads(path.read_text())
    saved = SavedGraph.model_validate(data)

    from src.core.component import Component

    classes = Component.registered_subclasses()

    nodes: dict[str, Node] = {}
    for nid, sn in saved.nodes.items():
        cls = classes.get(sn.type)
        if cls is None:
            raise ValueError(f"Unknown node type: {sn.type}")
        comp = cls.from_args(sn.init_args)
        nodes[nid] = Node(inner=comp, init_args=sn.init_args, x=sn.x, y=sn.y)

    edges = [
        Edge(
            source_node=se.source_node,
            source_slot=se.source_slot,
            target_node=se.target_node,
            target_slot=se.target_slot,
        )
        for se in saved.edges
    ]

    return Graph(edges=edges, nodes=nodes)


def close_project(config: AppConfig, current_graph: Graph | None = None) -> None:
    if current_graph is not None:
        from src.api.graph.run.service import stop_all

        stop_all(current_graph)
    config.current_project = None
    save_config(config)
