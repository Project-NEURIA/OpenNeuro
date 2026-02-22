from __future__ import annotations

from src.core.graph import Graph
from src.api.project.paths import graph_path


def save_graph(project_name: str, graph: Graph) -> None:
    path = graph_path(project_name)
    path.write_text(graph.model_dump_json(indent=2))
