from __future__ import annotations

import time

from src.api.graph.domain.graph import Graph
from src.api.metrics.dto import MetricsResponse


def collect(graph: Graph) -> MetricsResponse:
    nodes = dict(graph.nodes)
    return MetricsResponse(
        nodes={
            nid: node.inner.snapshot()
            for nid, node in nodes.items()
        },
        timestamp=time.time(),
    )
