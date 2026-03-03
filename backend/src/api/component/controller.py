from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import TypeAdapter

from src.api.component.dto import ComponentInfo
from src.api.component import service
from src.core.frames import *  # noqa: F403, F401

router = APIRouter(prefix="/component")


def _resolve_type(name: str) -> type:
    t = globals().get(name)
    if isinstance(t, type):
        return t
    raise ValueError(name)


def _type_name(t: type) -> str:
    import types

    if isinstance(t, types.UnionType):
        return "Union[" + ", ".join(_type_name(a) for a in t.__args__) + "]"
    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", None)
    if origin and args:
        name = origin.__name__
        inner = ", ".join(_type_name(a) for a in args)
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
        ui_inputs = cls.get_ui_input_types()
        ui_outputs = cls.get_ui_output_types()

        if not inputs:
            category = "source"
        elif not outputs:
            category = "sink"
        else:
            category = "conduit"

        result.append(
            ComponentInfo(
                name=name,
                category=category,
                init={k: TypeAdapter(v).json_schema() for k, v in init.items()},
                inputs={k: _type_name(v) for k, v in inputs.items()},
                outputs={k: _type_name(v) for k, v in outputs.items()},
                ui_inputs={k: _type_name(v) for k, v in ui_inputs.items()},
                ui_outputs={k: _type_name(v) for k, v in ui_outputs.items()},
            )
        )

    return result


@router.get("/is-type")
def is_type(name: str = Query()) -> bool:
    """Check if `name` resolves to a known type."""
    try:
        _resolve_type(name)
        return True
    except ValueError:
        return False


@router.get("/is-subtype")
def is_subtype(sub: str = Query(), sup: str = Query()) -> bool:
    """Check if `sub` is a subtype of `sup` using Python's issubclass()."""
    try:
        return issubclass(_resolve_type(sub), _resolve_type(sup))
    except ValueError:
        return False


@router.post("/{component_name}/options/{field_name}")
def get_config_options(
    component_name: str,
    field_name: str,
    values: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return runtime options for a dynamic config field."""
    classes = service.list_components()
    cls = classes.get(component_name)
    if cls is None:
        raise HTTPException(
            status_code=404, detail=f"Component not found: {component_name}"
        )
    result = cls.get_config_options(field_name, values)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Field not dynamic: {field_name}")
    return result
