from __future__ import annotations

from ...core.graph import Graph
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

    graph.edges.pop(node_id, None)
    for targets in graph.edges.values():
        if node_id in targets:
            targets.remove(node_id)

    del graph.nodes[node_id]


def list_edges(graph: Graph[Component]) -> list[tuple[str, str]]:
    return [
        (source, target)
        for source, targets in graph.edges.items()
        for target in targets
    ]


def create_edge(graph: Graph[Component], source_id: str, target_id: str) -> None:
    source = graph.nodes.get(source_id)
    if source is None:
        raise KeyError(f"Node not found: {source_id}")

    target = graph.nodes.get(target_id)
    if target is None:
        raise KeyError(f"Node not found: {target_id}")

    targets = graph.edges.setdefault(source_id, [])
    if target_id in targets:
        raise ValueError(f"Edge already exists: {source_id} -> {target_id}")

    targets.append(target_id)
    target.set_input_topics(*source.get_output_topics())


def start_all(graph: Graph[Component]) -> None:
    for node in graph.nodes.values():
        node.start()


def stop_all(graph: Graph[Component]) -> None:
    for node in graph.nodes.values():
        node.stop()


def delete_edge(graph: Graph[Component], source_id: str, target_id: str) -> None:
    targets = graph.edges.get(source_id)
    if targets is None or target_id not in targets:
        raise KeyError(f"Edge not found: {source_id} -> {target_id}")

    targets.remove(target_id)
