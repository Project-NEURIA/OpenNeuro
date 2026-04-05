from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import TypeAdapter

from src.api.component.dto import ComponentInfo
from src.api.component import service
from src.core.component import CompositeComponent, Tag
from src.core.config import PROJECTS_DIR
from src.core.frames import *  # noqa: F403, F401
from src.core.graph import Graph

router = APIRouter(prefix="/component")


def _resolve_type(name: str) -> type:
    builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    t = globals().get(name) or builtins.get(name)
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
    for type_, cls in classes.items():
        init = cls.get_init_types()
        inputs = cls._class_input_types()
        outputs = cls._class_output_types()
        ui_inputs = cls._class_ui_input_types()
        ui_outputs = cls._class_ui_output_types()

        result.append(
            ComponentInfo(
                type_=type_,
                description=getattr(cls, "description", ""),
                tags=cls.tags,
                init={k: TypeAdapter(v).json_schema() for k, v in init.items()},
                inputs={k: _type_name(v) for k, v in inputs.items()},
                outputs={k: _type_name(v) for k, v in outputs.items()},
                ui_inputs={k: _type_name(v) for k, v in ui_inputs.items()},
                ui_outputs={k: _type_name(v) for k, v in ui_outputs.items()},
            )
        )

    # Include saved projects as composite components
    if PROJECTS_DIR.exists():
        for d in sorted(PROJECTS_DIR.iterdir()):
            graph_path = d / "graph.json"
            if not d.is_dir() or not graph_path.exists():
                continue
            try:
                graph = Graph.model_validate(json.loads(graph_path.read_text()))
                comp = CompositeComponent(d.name, graph)
                result.append(
                    ComponentInfo(
                        type_=d.name,
                        description="",
                        tags=Tag(io={"conduit"}, functionality={"misc"}),
                        init={},
                        inputs={
                            k: _type_name(v) for k, v in comp.get_input_types().items()
                        },
                        outputs={
                            k: _type_name(v) for k, v in comp.get_output_types().items()
                        },
                        ui_inputs={},
                        ui_outputs={},
                        is_composite=True,
                        has_thumbnail=(d / "thumbnail.png").exists(),
                    )
                )
            except Exception:
                continue

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


@router.post("/subtype-pairs")
def subtype_pairs(names: list[str]) -> list[list[str]]:
    """Return all [sub, sup] pairs where sub is a subtype of sup."""
    result: list[list[str]] = []
    for a in names:
        for b in names:
            if a == b:
                continue
            try:
                if issubclass(_resolve_type(a), _resolve_type(b)):
                    result.append([a, b])
            except ValueError:
                pass
    return result


@router.post("/{component_name}/options")
def get_options(
    component_name: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Return all dynamic config options for a component."""
    classes = service.list_components()
    cls = classes.get(component_name)
    if cls is None:
        raise HTTPException(
            status_code=404, detail=f"Component not found: {component_name}"
        )
    return cls.get_options(values)
