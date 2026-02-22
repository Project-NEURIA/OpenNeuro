from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dep import get_current_project, get_manager
from src.api.graph.save import service
from src.core.graph import GraphManager

router = APIRouter(prefix="/graph")


@router.post("/save", status_code=204)
def save_graph(
    current_project: str = Depends(get_current_project),
    manager: GraphManager = Depends(get_manager),
) -> None:
    service.save_graph(current_project, manager.graph)
