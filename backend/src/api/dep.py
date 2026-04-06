from __future__ import annotations

from fastapi import Request

from src.api.relay.bridge import RelayBridge
from src.api.ui.bridge import UIChannelBridge
from src.core.graph import GraphManager


def get_manager(request: Request) -> GraphManager:
    return request.app.state.manager


def get_ui_bridge(request: Request) -> UIChannelBridge:
    return request.app.state.ui_bridge


def get_relay_bridge(request: Request) -> RelayBridge:
    return request.app.state.relay_bridge
