from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.core.graph import Graph


class NodeCreateRequest(BaseModel):
    type: str
    init_args: dict[str, Any] = {}


class SubgraphCreateRequest(BaseModel):
    node_ids: list[str]
    name: str = "Subgraph"


class NodeUpdateRequest(BaseModel):
    x: float
    y: float


class NodeResponse(BaseModel):
    id: str
    type: str
    is_composite: bool = False
    label: str | None = None
    status: str
    x: float
    y: float
    inputs: dict[str, str] | None = None
    outputs: dict[str, str] | None = None
    sub_graph: Graph | None = None
