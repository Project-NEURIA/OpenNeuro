from __future__ import annotations

import time

from src.api.metrics.dto import MetricsResponse
from src.core.graph import GraphManager


def collect(manager: GraphManager) -> MetricsResponse:
    return MetricsResponse(
        nodes={nid: comp.snapshot() for nid, comp in manager.components().items()},
        timestamp=time.time(),
    )
