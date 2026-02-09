from __future__ import annotations

from pydantic import BaseModel


class NodeCreateRequest(BaseModel):
    type: str
    id: str | None = None


class NodeResponse(BaseModel):
    id: str
    type: str
    status: str
