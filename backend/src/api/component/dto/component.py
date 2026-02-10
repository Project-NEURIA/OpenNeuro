from __future__ import annotations

from pydantic import BaseModel


class ComponentInfo(BaseModel):
    name: str
    category: str
    inputs: list[str]
