from __future__ import annotations

from fastapi import HTTPException, Request

from src.api.graph.domain.graph import Graph


def get_graph(request: Request) -> Graph:
    return request.app.state.graph


def get_current_project(request: Request) -> str:
    name: str | None = getattr(request.app.state, "current_project", None)
    if name is None:
        raise HTTPException(status_code=400, detail="No project is currently open")
    return name
