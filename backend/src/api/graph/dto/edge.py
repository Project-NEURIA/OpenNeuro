from __future__ import annotations

from pydantic import BaseModel


class EdgeCreateRequest(BaseModel):
    source_node: str
    source_slot: str
    target_node: str
    target_slot: str


class EdgeResponse(BaseModel):
    source_node: str
    source_slot: str
    target_node: str
    target_slot: str
