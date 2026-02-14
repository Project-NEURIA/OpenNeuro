from __future__ import annotations

from pydantic import BaseModel

from openneuro.core.component import ComponentSnapshot


class MetricsResponse(BaseModel):
    nodes: dict[str, ComponentSnapshot]
    timestamp: float
