from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dep import get_graph
from src.api.graph.domain.graph import Graph
from src.api.graph.edge.dto import EdgeCreateRequest, EdgeResponse
from src.api.graph.edge import service

router = APIRouter(prefix="/graph")


@router.get("/edges")
def list_edges(graph: Graph = Depends(get_graph)) -> list[EdgeResponse]:
    return [
        EdgeResponse(
            source_node=e.source_node,
            source_slot=e.source_slot,
            target_node=e.target_node,
            target_slot=e.target_slot,
        )
        for e in service.list_edges(graph)
    ]


@router.post("/edges", status_code=201)
def create_edge(
    req: EdgeCreateRequest, graph: Graph = Depends(get_graph)
) -> EdgeResponse:
    try:
        service.create_edge(
            graph, req.source_node, req.source_slot, req.target_node, req.target_slot
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EdgeResponse(
        source_node=req.source_node,
        source_slot=req.source_slot,
        target_node=req.target_node,
        target_slot=req.target_slot,
    )


@router.delete("/edges", status_code=204)
def delete_edge(req: EdgeCreateRequest, graph: Graph = Depends(get_graph)) -> None:
    try:
        service.delete_edge(
            graph, req.source_node, req.source_slot, req.target_node, req.target_slot
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
