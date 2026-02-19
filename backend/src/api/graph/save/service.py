from __future__ import annotations

from src.api.graph.domain.graph import Graph
from src.api.graph.save.dto import SavedEdge, SavedGraph, SavedNode
from src.api.project.paths import graph_path


def save_graph(
    project_name: str,
    graph: Graph,
) -> None:
    saved_nodes: dict[str, SavedNode] = {}
    for nid, node in graph.nodes.items():
        saved_nodes[nid] = SavedNode(
            type=type(node.inner).__name__,
            init_args=node.init_args,
            x=node.x,
            y=node.y,
        )

    saved_edges = [
        SavedEdge(
            source_node=e.source_node,
            source_slot=e.source_slot,
            target_node=e.target_node,
            target_slot=e.target_slot,
        )
        for e in graph.edges
    ]

    saved = SavedGraph(nodes=saved_nodes, edges=saved_edges)
    path = graph_path(project_name)
    path.write_text(saved.model_dump_json(indent=2))
