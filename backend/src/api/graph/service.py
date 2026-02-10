from __future__ import annotations

from ...core.graph import Graph, Edge
from ...core.component import Component


def list_nodes(graph: Graph[Component]) -> dict[str, Component]:
    return graph.nodes


def get_node(graph: Graph[Component], node_id: str) -> Component | None:
    return graph.nodes.get(node_id)


def create_node(graph: Graph[Component], node_type: str, node_id: str | None = None) -> Component:
    classes = Component.registered_subclasses()
    cls = classes.get(node_type)
    if cls is None:
        raise ValueError(f"Unknown node type: {node_type}")

    node_id = node_id or node_type
    if node_id in graph.nodes:
        return graph.nodes[node_id]

    node = cls()
    node.name = node_id
    graph.nodes[node_id] = node
    return node


def delete_node(graph: Graph[Component], node_id: str) -> None:
    node = graph.nodes.get(node_id)
    if node is None:
        return

    node.stop()

    # Collect target nodes that need stopping after edge removal
    targets: set[str] = set()
    for edge in graph.edges:
        if edge.source_node == node_id:
            targets.add(edge.target_node)

    graph.edges = [
        e for e in graph.edges
        if e.source_node != node_id and e.target_node != node_id
    ]

    for target_id in targets:
        if node := graph.nodes.get(target_id):
            node.stop()

    del graph.nodes[node_id]


def list_edges(graph: Graph[Component]) -> list[Edge]:
    return graph.edges


def create_edge(
    graph: Graph[Component],
    source_node: str,
    source_slot: int,
    target_node: str,
    target_slot: int,
) -> None:
    source = graph.nodes.get(source_node)
    if source is None:
        raise KeyError(f"Node not found: {source_node}")

    target = graph.nodes.get(target_node)
    if target is None:
        raise KeyError(f"Node not found: {target_node}")

    outputs = source.get_output_channels()
    if source_slot >= len(outputs):
        raise ValueError(f"source_slot {source_slot} out of range (node {source_node} has {len(outputs)} outputs)")

    input_types = type(target).get_input_types()
    if target_slot >= len(input_types):
        raise ValueError(f"target_slot {target_slot} out of range (node {target_node} has {len(input_types)} inputs)")

    edge = Edge(
        source_node=source_node,
        source_slot=source_slot,
        target_node=target_node,
        target_slot=target_slot,
    )

    if edge in graph.edges:
        raise ValueError(f"Edge already exists: {edge}")

    graph.edges.append(edge)
    _set_input_channels(target_node, graph)


def delete_edge(
    graph: Graph[Component],
    source_node: str,
    source_slot: int,
    target_node: str,
    target_slot: int,
) -> None:
    edge = Edge(
        source_node=source_node,
        source_slot=source_slot,
        target_node=target_node,
        target_slot=target_slot,
    )

    try:
        graph.edges.remove(edge)
        node = graph.nodes.get(target_node)
        if node:
            node.stop()
    except ValueError:
        raise KeyError(f"Edge not found: {edge}")


def _set_input_channels(target_node_id: str, graph: Graph[Component]) -> None:
    """Collect all incoming edges for a target node and call set_input_channels."""
    target = graph.nodes[target_node_id]
    incoming = sorted(
        (e for e in graph.edges if e.target_node == target_node_id),
        key=lambda e: e.target_slot,
    )

    if not incoming:
        return

    channels = []
    for edge in incoming:
        source = graph.nodes[edge.source_node]
        channels.append(source.get_output_channels()[edge.source_slot])

    target.set_input_channels(*channels)


def start_all(graph: Graph[Component]) -> None:
    for node in graph.nodes.values():
        node.start()


def stop_all(graph: Graph[Component]) -> None:
    for node in graph.nodes.values():
        node.stop()
