from __future__ import annotations

from fastapi import HTTPException, Request

from src.core.graph import GraphManager


def get_manager(request: Request) -> GraphManager:
    return request.app.state.manager


def get_current_project(request: Request) -> str:
    name: str | None = getattr(request.app.state, "current_project", None)
    if name is None:
        raise HTTPException(status_code=400, detail="No project is currently open")
    return name
