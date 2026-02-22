from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dep import get_manager
from src.api.graph.save import service
from src.core.config import AppConfig
from src.core.graph import GraphManager

router = APIRouter(prefix="/graph")


@router.post("/save", status_code=204)
def save_graph(
    manager: GraphManager = Depends(get_manager),
) -> None:
    service.save_graph(AppConfig.load_config().current_project, manager.graph)
