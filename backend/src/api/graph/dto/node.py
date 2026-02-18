from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class NodeCreateRequest(BaseModel):
    type: str
    config: dict[str, Any]


class NodeResponse(BaseModel):
    id: str
    type: str
    status: str
