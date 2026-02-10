from __future__ import annotations

import time

from ...core.graph import Graph
from ...core.component import Component
from .dto import MetricsResponse


def collect(graph: Graph[Component]) -> MetricsResponse:
    return MetricsResponse(
        nodes={nid: node.snapshot() for nid, node in graph.nodes.items()},
        timestamp=time.time(),
    )
