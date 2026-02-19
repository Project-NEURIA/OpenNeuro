from __future__ import annotations

from src.api.graph.domain.graph import Edge, Graph


def list_edges(graph: Graph) -> list[Edge]:
    return graph.edges


def create_edge(
    graph: Graph,
    source_node: str,
    source_slot: str,
    target_node: str,
    target_slot: str,
) -> None:
    source = graph.nodes.get(source_node)
    if source is None:
        raise KeyError(f"Node not found: {source_node}")

    target = graph.nodes.get(target_node)
    if target is None:
        raise KeyError(f"Node not found: {target_node}")

    # Validate source_slot exists in source's output types
    source_types = type(source.inner).get_output_types()
    if source_slot not in source_types:
        raise ValueError(
            f"source_slot '{source_slot}' not found in {source_node}'s outputs ({list(source_types)})"
        )

    # Validate target_slot exists in target's input types
    target_types = type(target.inner).get_input_types()
    if target_slot not in target_types:
        raise ValueError(
            f"target_slot '{target_slot}' not found in {target_node}'s inputs ({list(target_types)})"
        )

    edge = Edge(
        source_node=source_node,
        source_slot=source_slot,
        target_node=target_node,
        target_slot=target_slot,
    )

    if edge in graph.edges:
        raise ValueError(f"Edge already exists: {edge}")

    graph.edges.append(edge)


def delete_edge(
    graph: Graph,
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

    try:
        graph.edges.remove(edge)
        # Stop connected components
        for nid in (source_node, target_node):
            node = graph.nodes.get(nid)
            if node:
                node.inner.stop()
    except ValueError:
        raise KeyError(f"Edge not found: {edge}")
