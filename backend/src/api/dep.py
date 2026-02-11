from __future__ import annotations

from fastapi import Request

from src.api.graph.domain.graph import Graph
from src.core.component import Component


def get_graph(request: Request) -> Graph[Component]:
    return request.app.state.graph
