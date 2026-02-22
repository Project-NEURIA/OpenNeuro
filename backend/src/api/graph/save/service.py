from __future__ import annotations

from src.core.config import PROJECTS_DIR
from src.core.graph import Graph


def save_graph(project_name: str, graph: Graph) -> None:
    path = PROJECTS_DIR / project_name / "graph.json"
    path.write_text(graph.model_dump_json(indent=2))
