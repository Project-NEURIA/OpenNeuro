from __future__ import annotations

from fastapi import Request

from src.core.graph import GraphManager


def get_manager(request: Request) -> GraphManager:
    return request.app.state.manager
