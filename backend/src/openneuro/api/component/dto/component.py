from __future__ import annotations

from pydantic import BaseModel


class ComponentInfo(BaseModel):
    name: str
    category: str
    inputs: list[str]
    input_names: list[str]  # Human-readable names for each input slot
    outputs: list[str]
