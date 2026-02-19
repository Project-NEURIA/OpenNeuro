from __future__ import annotations

import inspect
from typing import Any, Union

from src.api.graph.domain.graph import Graph, Node
from src.api.graph.save.service import _auto_save
from src.core.component import Component


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
