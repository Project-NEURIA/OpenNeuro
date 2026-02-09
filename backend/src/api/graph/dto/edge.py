from __future__ import annotations

from pydantic import BaseModel


class EdgeCreateRequest(BaseModel):
    source: str
    target: str


class EdgeResponse(BaseModel):
    source: str
    target: str
