from __future__ import annotations

from fastapi import Request

from src.api.graph.domain.graph import Graph


def get_graph(request: Request) -> Graph:
    return request.app.state.graph
