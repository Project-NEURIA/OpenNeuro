from __future__ import annotations

import time

from ...core.graph import Graph
from ...core.component import Component, ComponentMetadata
from ...core.topic import NOTOPIC, Topic, TopicSnapshot


def collect(graph: Graph[Component]) -> dict:
    nodes: dict[str, ComponentMetadata] = {}
    topics: dict[str, TopicSnapshot] = {}

    for node_id, node in graph.nodes.items():
        nodes[node_id] = node.metadata()

        for topic in node.get_output_topics():
            if topic is NOTOPIC:
                continue
            snap = topic.snapshot()
            topics[snap.name] = snap

    return {
        "nodes": nodes,
        "topics": topics,
        "timestamp": time.time(),
    }
