from __future__ import annotations

import uuid
from typing import Any

from src.api.graph.domain.graph import Graph, Node
from src.api.graph.node.dto import NodeUpdateRequest
from src.core.component import Component


def list_nodes(graph: Graph) -> dict[str, Node]:
    return graph.nodes


def get_node(graph: Graph, node_id: str) -> Node | None:
    return graph.nodes.get(node_id)


def create_node(
    graph: Graph, node_type: str, init_args: dict[str, Any]
) -> tuple[str, Node]:
    classes = Component.registered_subclasses()
    cls = classes.get(node_type)
    if cls is None:
        raise ValueError(f"Unknown node type: {node_type}")
    comp = cls.from_args(init_args)
    node_id = str(uuid.uuid4())
    node = Node(inner=comp, init_args=init_args)
    graph.nodes[node_id] = node
    return node_id, node


def update_node(graph: Graph, node_id: str, req: NodeUpdateRequest) -> Node | None:
    node = graph.nodes.get(node_id)
    if node is None:
        return None
    if req.x is not None:
        node.x = req.x
    if req.y is not None:
        node.y = req.y
    return node


def delete_node(graph: Graph, node_id: str) -> None:
    node = graph.nodes.get(node_id)
    if node is None:
        return

    node.inner.stop()

    # Collect connected components that need stopping
    affected: set[str] = set()
    for edge in graph.edges:
        if edge.source_node == node_id:
            affected.add(edge.target_node)
        if edge.target_node == node_id:
            affected.add(edge.source_node)

    graph.edges = [
        e for e in graph.edges if e.source_node != node_id and e.target_node != node_id
    ]

    for affected_id in affected:
        affected_node = graph.nodes.get(affected_id)
        if affected_node:
            affected_node.inner.stop()

    del graph.nodes[node_id]
