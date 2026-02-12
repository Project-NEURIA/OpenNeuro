from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from src.core.component import Component


class Node(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Component[..., Any]
    x: float = 0.0
    y: float = 0.0


class Edge(BaseModel):
    source_node: str
    source_slot: str
    target_node: str
    target_slot: str


class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    edges: list[Edge]
    nodes: dict[str, Node]
