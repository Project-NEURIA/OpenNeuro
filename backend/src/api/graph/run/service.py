from __future__ import annotations

from src.api.ui.bridge import UIChannelBridge
from src.core.graph import GraphManager


def start_all(manager: GraphManager, ui_bridge: UIChannelBridge) -> None:
    recv_overrides, send_overrides = ui_bridge.wire(manager)
    manager.run(recv_overrides, send_overrides)


def stop_all(manager: GraphManager) -> None:
    manager.stop()
