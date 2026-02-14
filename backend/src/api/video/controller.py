from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.graph.domain.graph import Graph
from src.core.sink.video_stream import VideoStream

router = APIRouter(prefix="/video")


@router.websocket("/ws/{node_id}")
async def video_stream(websocket: WebSocket, node_id: str) -> None:
    graph: Graph = websocket.app.state.graph
    node = graph.nodes.get(node_id)
    if node is None or not isinstance(node.inner, VideoStream):
        await websocket.close(code=4004, reason="Not a VideoStream node")
        return

    sink: VideoStream = node.inner
    await websocket.accept()

    try:
        while True:
            frame = await asyncio.to_thread(sink.wait_for_frame, 1.0)
            if frame is not None:
                await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        pass
