from __future__ import annotations

from pydantic import BaseModel


class ComponentInfo(BaseModel):
    name: str
    category: str
    init: dict[str, str]
    inputs: dict[str, str]
    outputs: dict[str, str]
