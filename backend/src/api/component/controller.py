from __future__ import annotations

from fastapi import APIRouter, Query
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
