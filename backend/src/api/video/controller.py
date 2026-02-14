from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from src.api.dep import get_graph
from src.api.graph.domain.graph import Graph
from src.core.sink.video_stream import VideoStream

router = APIRouter(prefix="/video")


@router.get("/{node_id}/frame")
def get_frame(node_id: str, graph: Graph = Depends(get_graph)) -> Response:
    node = graph.nodes.get(node_id)
    if node is None or not isinstance(node.inner, VideoStream):
        raise HTTPException(status_code=404, detail="Not a VideoStream node")

    frame = node.inner.latest_frame
    if frame is None:
        raise HTTPException(status_code=204)

    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
