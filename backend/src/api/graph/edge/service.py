from __future__ import annotations

from src.core.graph import Edge
from src.core.graph import GraphManager


def list_edges(manager: GraphManager) -> list[Edge]:
    return manager.graph.edges


def create_edge(
    manager: GraphManager,
    source_node: str,
    source_slot: str,
    target_node: str,
    target_slot: str,
) -> None:
    if manager.get_node(source_node) is None:
        raise KeyError(f"Node not found: {source_node}")
    if manager.get_node(target_node) is None:
        raise KeyError(f"Node not found: {target_node}")

    source_slots = manager.get_node_output(source_node)
    if source_slot not in source_slots:
        raise ValueError(
            f"source_slot '{source_slot}' not found in {source_node}'s outputs ({list(source_slots)})"
        )

    target_slots = manager.get_node_input(target_node)
    if target_slot not in target_slots:
        raise ValueError(
            f"target_slot '{target_slot}' not found in {target_node}'s inputs ({list(target_slots)})"
        )

    edge = Edge(
        source_node=source_node,
        source_slot=source_slot,
        target_node=target_node,
        target_slot=target_slot,
    )

    if edge in manager.graph.edges:
        raise ValueError(f"Edge already exists: {edge}")

    manager.add_edge(edge)


def delete_edge(
    manager: GraphManager,
    source_node: str,
    source_slot: str,
    target_node: str,
    target_slot: str,
) -> None:
    edge = Edge(
        source_node=source_node,
        source_slot=source_slot,
        target_node=target_node,
        target_slot=target_slot,
    )

    if edge not in manager.graph.edges:
        raise KeyError(f"Edge not found: {edge}")

    manager.delete_edge(edge)
