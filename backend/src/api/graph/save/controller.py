from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dep import get_current_project, get_graph
from src.api.graph.domain.graph import Graph
from src.api.graph.save import service

router = APIRouter(prefix="/graph")


@router.post("/save", status_code=204)
def save_graph(
    current_project: str = Depends(get_current_project),
    graph: Graph = Depends(get_graph),
) -> None:
    service.save_graph(current_project, graph)
