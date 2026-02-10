from __future__ import annotations

from pydantic import BaseModel

from ....core.component import ComponentMetadata
from ....core.topic import TopicSnapshot


class MetricsResponse(BaseModel):
    nodes: dict[str, ComponentMetadata]
    topics: dict[str, TopicSnapshot]
    timestamp: float
