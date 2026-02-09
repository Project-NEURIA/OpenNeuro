from __future__ import annotations

import time

from ...core.graph import Graph
from ...core.node import Node, NodeMetadata
from ...core.topic import Topic, TopicSnapshot


def collect(graph: Graph[Node]) -> dict:
    nodes: dict[str, NodeMetadata] = {}
    topics: dict[str, TopicSnapshot] = {}

    for node_id, node in graph.nodes.items():
        nodes[node_id] = node.metadata()

        for topic in node.topics():
            if not isinstance(topic, Topic):
                continue
            snap = topic.snapshot()
            topics[snap.name] = snap

    return {
        "nodes": nodes,
        "topics": topics,
        "timestamp": time.time(),
    }
