from __future__ import annotations

from pydantic import BaseModel

from ....core.node import NodeMetadata
from ....core.topic import TopicSnapshot


class MetricsResponse(BaseModel):
    nodes: dict[str, NodeMetadata]
    topics: dict[str, TopicSnapshot]
    timestamp: float
