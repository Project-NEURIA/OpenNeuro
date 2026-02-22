from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from src.api.dep import get_manager
from src.core.graph import GraphManager
from src.core.sink.video_stream import VideoStream

router = APIRouter(prefix="/video")


@router.get("/{node_id}/frame")
def get_frame(node_id: str, manager: GraphManager = Depends(get_manager)) -> Response:
    try:
        comp = manager.component(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Not a VideoStream node")
    if not isinstance(comp, VideoStream):
        raise HTTPException(status_code=404, detail="Not a VideoStream node")

    frame = comp.latest_frame
    if frame is None:
        raise HTTPException(status_code=204)

    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
