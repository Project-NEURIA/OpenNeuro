from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.graph.domain.graph import Graph, Node
from src.api.dep import get_graph
from src.api.graph.dto import NodeCreateRequest, NodeResponse, EdgeCreateRequest, EdgeResponse
from src.api.graph import service

router = APIRouter(prefix="/graph")


def _node_response(node_id: str, node: Node) -> NodeResponse:
    return NodeResponse(id=node_id, type=type(node.inner).__name__, status=node.inner.status.value)


@router.get("/nodes")
def list_nodes(graph: Graph = Depends(get_graph)) -> list[NodeResponse]:
    return [_node_response(nid, node) for nid, node in service.list_nodes(graph).items()]


@router.get("/nodes/{node_id}")
def get_node(node_id: str, graph: Graph = Depends(get_graph)) -> NodeResponse:
    node = service.get_node(graph, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return _node_response(node_id, node)


@router.post("/nodes", status_code=201)
def create_node(req: NodeCreateRequest, graph: Graph = Depends(get_graph)) -> NodeResponse:
    try:
        node = service.create_node(graph, req.type, req.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    node_id = req.id or req.type
    return _node_response(node_id, node)


@router.delete("/nodes/{node_id}", status_code=204)
def delete_node(node_id: str, graph: Graph = Depends(get_graph)) -> None:
    service.delete_node(graph, node_id)


@router.post("/start", status_code=204)
def start_all(graph: Graph = Depends(get_graph)) -> None:
    service.start_all(graph)


@router.post("/stop", status_code=204)
def stop_all(graph: Graph = Depends(get_graph)) -> None:
    service.stop_all(graph)


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
def create_edge(req: EdgeCreateRequest, graph: Graph = Depends(get_graph)) -> EdgeResponse:
    try:
        service.create_edge(graph, req.source_node, req.source_slot, req.target_node, req.target_slot)
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
        service.delete_edge(graph, req.source_node, req.source_slot, req.target_node, req.target_slot)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
