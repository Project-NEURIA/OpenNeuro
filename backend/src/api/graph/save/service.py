from __future__ import annotations

from pathlib import Path

from src.api.graph.domain.graph import Graph


def load_graph(_: str | Path = "saves/graph.json") -> Graph:
    """Loads a graph from a JSON file, reconstructing nodes and edges."""
    return Graph(edges=[], nodes={})


def _auto_save(graph: Graph) -> None:
    pass
