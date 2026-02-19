from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dep import get_graph
from src.api.graph.domain.graph import Graph
from src.api.graph.run import service

router = APIRouter(prefix="/graph")


@router.post("/start", status_code=204)
def start_all(graph: Graph = Depends(get_graph)) -> None:
    service.start_all(graph)


@router.post("/stop", status_code=204)
def stop_all(graph: Graph = Depends(get_graph)) -> None:
    service.stop_all(graph)
