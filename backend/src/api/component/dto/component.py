from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ComponentInfo(BaseModel):
    name: str
    category: str
    init: dict[str, Any]
    inputs: dict[str, str]
    outputs: dict[str, str]
