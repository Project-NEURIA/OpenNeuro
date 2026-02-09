from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ...core.graph import Graph
from ...core.node import Node
from .dto import NodeCreateRequest, NodeResponse, EdgeCreateRequest, EdgeResponse
from . import service

router = APIRouter(prefix="/graph")


def get_graph(request: Request) -> Graph[Node]:
    return request.app.state.graph


@router.get("/nodes")
def list_nodes(graph: Graph[Node] = Depends(get_graph)) -> list[NodeResponse]:
    return [
        NodeResponse(id=node_id, type=type(node).__name__, status=node.status.value)
        for node_id, node in service.list_nodes(graph).items()
    ]


@router.get("/nodes/{node_id}")
def get_node(node_id: str, graph: Graph[Node] = Depends(get_graph)) -> NodeResponse:
    node = service.get_node(graph, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return NodeResponse(id=node_id, type=type(node).__name__, status=node.status.value)


@router.post("/nodes", status_code=201)
def create_node(req: NodeCreateRequest, graph: Graph[Node] = Depends(get_graph)) -> NodeResponse:
    try:
        node = service.create_node(graph, req.type, req.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return NodeResponse(id=node.name, type=type(node).__name__, status=node.status.value)


@router.delete("/nodes/{node_id}", status_code=204)
def delete_node(node_id: str, graph: Graph[Node] = Depends(get_graph)) -> None:
    try:
        service.delete_node(graph, node_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/start", status_code=204)
def start_all(graph: Graph[Node] = Depends(get_graph)) -> None:
    service.start_all(graph)


@router.post("/stop", status_code=204)
def stop_all(graph: Graph[Node] = Depends(get_graph)) -> None:
    service.stop_all(graph)


@router.get("/edges")
def list_edges(graph: Graph[Node] = Depends(get_graph)) -> list[EdgeResponse]:
    return [
        EdgeResponse(source=source, target=target)
        for source, target in service.list_edges(graph)
    ]


@router.post("/edges", status_code=201)
def create_edge(req: EdgeCreateRequest, graph: Graph[Node] = Depends(get_graph)) -> EdgeResponse:
    try:
        service.create_edge(graph, req.source, req.target)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return EdgeResponse(source=req.source, target=req.target)


@router.delete("/edges", status_code=204)
def delete_edge(req: EdgeCreateRequest, graph: Graph[Node] = Depends(get_graph)) -> None:
    try:
        service.delete_edge(graph, req.source, req.target)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
