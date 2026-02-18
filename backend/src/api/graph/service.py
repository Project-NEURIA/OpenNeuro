from __future__ import annotations

from typing import Any, Union

from src.api.graph.domain.graph import Graph, Node, Edge
from src.core.channel import Channel
from src.core.component import Component
from pathlib import Path


import inspect


def load_graph(_: str | Path = "saves/graph.json") -> Graph:
    """Loads a graph from a JSON file, reconstructing nodes and edges."""
    return Graph(edges=[],nodes={})


def _auto_save(graph: Graph) -> None:
    pass


def list_nodes(graph: Graph) -> dict[str, Node]:
    return graph.nodes


def get_node(graph: Graph, node_id: str) -> Node | None:
    return graph.nodes.get(node_id)


def _resolve_annotation(cls: type, annotation: Any) -> type:
    """Resolve a string or Union annotation to a concrete type."""
    if isinstance(annotation, str):
        name = annotation.split("|")[0].strip() if "|" in annotation else annotation
        module = inspect.getmodule(cls)
        if module:
            return getattr(module, name)
    origin = getattr(annotation, "__origin__", None)
    if origin is Union:
        return next(a for a in annotation.__args__ if a is not type(None))
    return annotation


def create_node(
    graph: Graph, node_type: str, config: dict[str, Any]
) -> tuple[str, Node]:
    classes = Component.registered_subclasses()
    cls = classes.get(node_type)
    if cls is None:
        raise ValueError(f"Unknown node type: {node_type}")

    if config is not None:
        sig = inspect.signature(cls.__init__)
        kwargs: dict[str, Any] = {}
        for param_name, param_value in config.items():
            param = sig.parameters.get(param_name)
            if param is None or param.annotation is inspect.Parameter.empty:
                continue
            param_cls = _resolve_annotation(cls, param.annotation)
            if isinstance(param_value, dict) and hasattr(param_cls, "model_validate"):
                kwargs[param_name] = param_cls.model_validate(param_value)
            else:
                kwargs[param_name] = param_value
        comp = cls(**kwargs) if kwargs else cls()  # type: ignore
    else:
        comp = cls()

    node_id = str(id(comp))
    comp.name = node_type
    node = Node(inner=comp, config=config)
    graph.nodes[node_id] = node
    _auto_save(graph)
    return node_id, node


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
    _auto_save(graph)


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
    _auto_save(graph)


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
        _auto_save(graph)
    except ValueError:
        raise KeyError(f"Edge not found: {edge}")


def start_all(graph: Graph) -> None:
    comp_inputs: dict[str, dict[str, Channel[Any]]] = {nid: {} for nid in graph.nodes}

    for edge in graph.edges:
        source = graph.nodes[edge.source_node].inner
        channel = source.get_output_channels()[edge.source_slot]
        comp_inputs[edge.target_node][edge.target_slot] = channel

    for node_id, node in graph.nodes.items():
        node.inner.start(**comp_inputs[node_id])


def stop_all(graph: Graph) -> None:
    for node in graph.nodes.values():
        node.inner.stop()
