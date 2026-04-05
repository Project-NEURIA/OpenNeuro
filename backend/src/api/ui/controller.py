from __future__ import annotations

from fastapi import APIRouter, WebSocket

from src.api.ui.bridge import UIChannelBridge

router = APIRouter(prefix="/ui")


@router.websocket("/ws")
async def ui_ws(ws: WebSocket) -> None:
    bridge: UIChannelBridge = ws.app.state.ui_bridge
    await ws.accept()
    await bridge.run(ws)
