from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dep import get_manager
from src.core.graph import Node
from src.api.graph.node.dto import NodeCreateRequest, NodeResponse, NodeUpdateRequest
from src.api.graph.node import service
from src.core.graph import GraphManager

router = APIRouter(prefix="/graph")


def _node_response(node_id: str, node: Node, manager: GraphManager) -> NodeResponse:
    comp = manager.component(node_id)
    return NodeResponse(
        id=node_id,
        type=node.type,
        status=comp.status.value,
        x=node.x,
        y=node.y,
    )


@router.get("/nodes")
def list_nodes(manager: GraphManager = Depends(get_manager)) -> list[NodeResponse]:
    return [
        _node_response(nid, node, manager)
        for nid, node in service.list_nodes(manager).items()
    ]


@router.get("/nodes/{node_id}")
def get_node(
    node_id: str, manager: GraphManager = Depends(get_manager)
) -> NodeResponse:
    node = service.get_node(manager, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return _node_response(node_id, node, manager)


@router.post("/nodes", status_code=201)
def create_node(
    req: NodeCreateRequest, manager: GraphManager = Depends(get_manager)
) -> NodeResponse:
    try:
        node_id, node = service.create_node(manager, req.type, req.init_args)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _node_response(node_id, node, manager)


@router.patch("/nodes/{node_id}")
def update_node(
    node_id: str, req: NodeUpdateRequest, manager: GraphManager = Depends(get_manager)
) -> NodeResponse:
    node = service.update_node(manager, node_id, req)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return _node_response(node_id, node, manager)


@router.delete("/nodes/{node_id}", status_code=204)
def delete_node(node_id: str, manager: GraphManager = Depends(get_manager)) -> None:
    service.delete_node(manager, node_id)
