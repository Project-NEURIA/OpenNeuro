from __future__ import annotations

from typing import Any

from src.core.component import Component, PrimitiveComponent


def list_components() -> dict[str, type[PrimitiveComponent[Any, Any]]]:
    return Component.registered_subclasses()
