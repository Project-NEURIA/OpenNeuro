from __future__ import annotations

from fastapi import APIRouter
from pydantic import TypeAdapter

from src.api.component.dto import ComponentInfo
from src.api.component import service

router = APIRouter(prefix="/component")


def _type_name(t: type) -> str:
    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", None)
    if origin and args:
        name = origin.__name__
        inner = ", ".join(a.__name__ if hasattr(a, "__name__") else str(a) for a in args)
        return f"{name}[{inner}]"
    return getattr(t, "__name__", str(t))


@router.get("")
def list_components() -> list[ComponentInfo]:
    classes = service.list_components()
    result = []
    for name, cls in classes.items():
        init = cls.get_init_types()
        inputs = cls.get_input_types()
        outputs = cls.get_output_types()

        if not inputs:
            category = "source"
        elif not outputs:
            category = "sink"
        else:
            category = "conduit"

        result.append(ComponentInfo(
            name=name,
            category=category,
            init={k: TypeAdapter(v).json_schema() for k, v in init.items()},
            inputs={k: _type_name(v) for k, v in inputs.items()},
            outputs={k: _type_name(v) for k, v in outputs.items()},
        ))

    return result
