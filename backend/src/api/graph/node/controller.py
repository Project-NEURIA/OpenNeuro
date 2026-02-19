from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dep import get_graph
from src.api.graph.domain.graph import Graph, Node
from src.api.graph.node.dto import NodeCreateRequest, NodeResponse, NodeUpdateRequest
from src.api.graph.node import service

router = APIRouter(prefix="/graph")


def _node_response(node_id: str, node: Node) -> NodeResponse:
    return NodeResponse(
        id=node_id,
        type=type(node.inner).__name__,
        status=node.inner.status.value,
        x=node.x,
        y=node.y,
    )


@router.get("/nodes")
def list_nodes(graph: Graph = Depends(get_graph)) -> list[NodeResponse]:
    return [
        _node_response(nid, node) for nid, node in service.list_nodes(graph).items()
    ]


@router.get("/nodes/{node_id}")
def get_node(node_id: str, graph: Graph = Depends(get_graph)) -> NodeResponse:
    node = service.get_node(graph, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return _node_response(node_id, node)


@router.post("/nodes", status_code=201)
def create_node(
    req: NodeCreateRequest, graph: Graph = Depends(get_graph)
) -> NodeResponse:
    try:
        node_id, node = service.create_node(graph, req.type, req.init_args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _node_response(node_id, node)


@router.patch("/nodes/{node_id}")
def update_node(
    node_id: str, req: NodeUpdateRequest, graph: Graph = Depends(get_graph)
) -> NodeResponse:
    node = service.update_node(graph, node_id, req)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return _node_response(node_id, node)


@router.delete("/nodes/{node_id}", status_code=204)
def delete_node(node_id: str, graph: Graph = Depends(get_graph)) -> None:
    service.delete_node(graph, node_id)
