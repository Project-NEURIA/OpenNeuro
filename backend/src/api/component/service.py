from __future__ import annotations

from typing import Any

from src.core.component import PrimitiveComponent


def list_components() -> dict[str, type[PrimitiveComponent[Any, Any]]]:
    return PrimitiveComponent.registered_subclasses()
