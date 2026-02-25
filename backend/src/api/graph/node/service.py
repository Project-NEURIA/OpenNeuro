from __future__ import annotations

from typing import Any

from src.core.graph import Node
from src.api.graph.node.dto import NodeUpdateRequest
from src.core.graph import GraphManager


def list_nodes(manager: GraphManager) -> dict[str, Node]:
    return manager.graph.nodes


def get_node(manager: GraphManager, node_id: str) -> Node | None:
    return manager.get_node(node_id)


def create_node(
    manager: GraphManager, node_type: str, init_args: dict[str, Any]
) -> tuple[str, Node]:
    return manager.add_node(node_type, init_args)


def update_node(
    manager: GraphManager, node_id: str, req: NodeUpdateRequest
) -> Node | None:
    return manager.update_node(node_id, x=req.x, y=req.y)


def delete_node(manager: GraphManager, node_id: str) -> None:
    manager.delete_node(node_id)
