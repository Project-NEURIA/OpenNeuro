from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class NodeCreateRequest(BaseModel):
    type: str
    init_args: dict[str, Any]


class NodeUpdateRequest(BaseModel):
    x: float | None = None
    y: float | None = None


class NodeResponse(BaseModel):
    id: str
    type: str
    status: str
    x: float
    y: float
