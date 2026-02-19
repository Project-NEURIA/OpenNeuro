from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SavedNode(BaseModel):
    type: str
    init_args: dict[str, Any]
    x: float
    y: float


class SavedEdge(BaseModel):
    source_node: str
    source_slot: str
    target_node: str
    target_slot: str


class SavedGraph(BaseModel):
    nodes: dict[str, SavedNode]
    edges: list[SavedEdge]
