from __future__ import annotations

import time

from openneuro.api.graph.domain.graph import Graph
from openneuro.core.component import Component
from openneuro.api.metrics.dto import MetricsResponse


def collect(graph: Graph[Component]) -> MetricsResponse:
    return MetricsResponse(
        nodes={nid: node.snapshot() for nid, node in graph.nodes.items()},
        timestamp=time.time(),
    )
