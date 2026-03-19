from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.core.component import Tag


class ComponentInfo(BaseModel):
    type_: str
    description: str | None = None
    tags: Tag
    init: dict[str, Any]
    inputs: dict[str, str]
    outputs: dict[str, str]
    ui_inputs: dict[str, str]
    ui_outputs: dict[str, str]
