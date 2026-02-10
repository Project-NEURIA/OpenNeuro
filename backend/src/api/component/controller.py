from __future__ import annotations

from fastapi import APIRouter

from .dto import ComponentInfo
from . import service

router = APIRouter(prefix="/component")


def _type_str(t: type) -> str:
    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", ())
    if origin is not None and args:
        name = getattr(origin, "__name__", str(origin))
        inner = ", ".join(_type_str(a) for a in args)
        return f"{name}[{inner}]"
    return getattr(t, "__name__", str(t))


@router.get("")
def list_components() -> list[ComponentInfo]:
    classes = service.list_components()
    result = []
    for name, cls in classes.items():
        inputs, outputs = cls.get_sig()

        if not inputs:
            category = "source"
        elif not outputs:
            category = "sink"
        else:
            category = "conduit"

        result.append(ComponentInfo(
            name=name,
            category=category,
            inputs=[_type_str(t) for t in inputs],
            outputs=[_type_str(t) for t in outputs],
        ))
    return result
