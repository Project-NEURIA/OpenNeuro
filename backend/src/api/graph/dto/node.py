from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class NodeCreateRequest(BaseModel):
    type: str
    config: dict[str, Any] | None = None


class NodeResponse(BaseModel):
    id: str
    type: str
    status: str
