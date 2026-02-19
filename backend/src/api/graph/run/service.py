from __future__ import annotations

from typing import Any

from src.api.graph.domain.graph import Graph
from src.core.channel import Channel


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
