from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dep import get_manager, get_ui_bridge
from src.api.graph.run import service
from src.api.ui.bridge import UIChannelBridge
from src.core.graph import GraphManager

router = APIRouter(prefix="/graph")


@router.post("/start", status_code=204)
def start_all(
    manager: GraphManager = Depends(get_manager),
    ui_bridge: UIChannelBridge = Depends(get_ui_bridge),
) -> None:
    service.start_all(manager, ui_bridge)


@router.post("/stop", status_code=204)
def stop_all(manager: GraphManager = Depends(get_manager)) -> None:
    service.stop_all(manager)
