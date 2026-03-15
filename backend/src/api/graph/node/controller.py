from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dep import get_manager
from src.api.graph.node.dto import (
    NodeCreateRequest,
    NodeResponse,
    NodeUpdateRequest,
    SubgraphCreateRequest,
)
from src.api.component.controller import _type_name
from src.api.graph.node import service
from src.core.component import CompositeComponent
from src.core.graph import GraphManager, Node

router = APIRouter(prefix="/graph")


def _node_response(node_id: str, node: Node, manager: GraphManager) -> NodeResponse:
    comp = manager.component(node_id)
    inputs: dict[str, str] | None = None
    outputs: dict[str, str] | None = None
    if isinstance(comp, CompositeComponent):
        inputs = {k: _type_name(v) for k, v in comp.get_input_types().items()}
        outputs = {k: _type_name(v) for k, v in comp.get_output_types().items()}
    return NodeResponse(
        id=node_id,
        type=node.type,
        is_composite=node.is_composite,
        status=comp.status.value,
        x=node.x,
        y=node.y,
        inputs=inputs,
        outputs=outputs,
        sub_graph=node.sub_graph,
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


@router.post("/subgraph", status_code=201)
def create_subgraph(
    req: SubgraphCreateRequest, manager: GraphManager = Depends(get_manager)
) -> NodeResponse:
    try:
        node_id, node = service.create_subgraph(manager, req.node_ids, req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _node_response(node_id, node, manager)


@router.post("/ungroup/{node_id}")
def ungroup_node(
    node_id: str, manager: GraphManager = Depends(get_manager)
) -> list[NodeResponse]:
    try:
        service.ungroup(manager, node_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        _node_response(nid, node, manager)
        for nid, node in service.list_nodes(manager).items()
    ]
