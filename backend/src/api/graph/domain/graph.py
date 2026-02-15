from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from src.core.component import Component


import json
from pathlib import Path


class Node(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Component[..., Any]
    config: dict[..., ...]
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

    def save_to_file(self, path: str | Path = "saves/graph.json"):
        """Saves current graph state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": {
                node_id: {
                    "type": type(node.inner).__name__,
                    "x": node.x,
                    "y": node.y,
                    "config": node.inner.config.to_dict(),
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source_node": edge.source_node,
                    "source_slot": edge.source_slot,
                    "target_node": edge.target_node,
                    "target_slot": edge.target_slot,
                }
                for edge in self.edges
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
