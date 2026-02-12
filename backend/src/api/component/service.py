from __future__ import annotations

from typing import Any

from src.core.component import Component


def list_components() -> dict[str, type[Component[..., Any]]]:
    return Component.registered_subclasses()
