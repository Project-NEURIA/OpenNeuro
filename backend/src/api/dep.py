from __future__ import annotations

from fastapi import Request

from ..core.graph import Graph
from ..core.component import Component


def get_graph(request: Request) -> Graph[Component]:
    return request.app.state.graph
