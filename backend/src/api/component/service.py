from __future__ import annotations

from src.core.component import Component


def list_components() -> dict[str, type[Component]]:
    return Component.registered_subclasses()
