from __future__ import annotations

from fastapi import Request

from openneuro.api.graph.domain.graph import Graph
from openneuro.core.component import Component


def get_graph(request: Request) -> Graph[Component]:
    return request.app.state.graph
