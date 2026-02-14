from __future__ import annotations

from fastapi import APIRouter

from openneuro.api.component.dto import ComponentInfo
from openneuro.api.component import service

router = APIRouter(prefix="/component")


@router.get("")
def list_components() -> list[ComponentInfo]:
    classes = service.list_components()
    result = []
    for name, cls in classes.items():
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
            inputs=[getattr(t, "__name__", str(t)) for t in inputs],
            input_names=cls.get_input_names(),
            outputs=[getattr(t, "__name__", str(t)) for t in outputs],
        ))
    return result
