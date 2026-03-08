from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NodeCreateRequest(BaseModel):
    type: str
    init_args: dict[str, Any] = {}


class NodeUpdateRequest(BaseModel):
    x: float
    y: float


class NodeResponse(BaseModel):
    id: str
    type: str
    status: str
    x: float
    y: float
    init_args: dict[str, Any] = Field(default_factory=dict)


class CharacterCardUpdateRequest(BaseModel):
    character_card: str


class NodeInitArgsUpdateRequest(BaseModel):
    init_args: dict[str, Any]
