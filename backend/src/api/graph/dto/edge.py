from __future__ import annotations

from pydantic import BaseModel


class EdgeCreateRequest(BaseModel):
    source_node: str
    source_slot: int
    target_node: str
    target_slot: int


class EdgeResponse(BaseModel):
    source_node: str
    source_slot: int
    target_node: str
    target_slot: int
