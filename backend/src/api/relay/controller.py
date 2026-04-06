from __future__ import annotations

from fastapi import APIRouter, WebSocket

from src.api.relay.bridge import RelayBridge

router = APIRouter(prefix="/relay")


@router.websocket("/ws")
async def relay_ws(ws: WebSocket) -> None:
    bridge: RelayBridge = ws.app.state.relay_bridge
    await ws.accept()
    await bridge.run(ws)
