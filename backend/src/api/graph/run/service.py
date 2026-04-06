from __future__ import annotations

from src.api.relay.bridge import RelayBridge
from src.api.ui.bridge import UIChannelBridge
from src.core.graph import GraphManager
from src.core.transport import (
    RelayOscTransport,
    RelayPoseTransport,
    RelayVideoSourceTransport,
)


def _inject_relay_transports(
    manager: GraphManager, relay_bridge: RelayBridge
) -> None:
    """Replace local transports with relay transports for PCVR-facing components."""
    from src.lib.misc.osc_chatbox import OSCChatbox
    from src.lib.motion.osc_face import OSCFace
    from src.lib.motion.openvr_movement import OpenVRMovementSink
    from src.lib.motion.vrchat import VRChatVideo

    for comp in manager.components().values():
        if isinstance(comp, (OSCChatbox, OSCFace)):
            comp._client = RelayOscTransport(relay_bridge)
        elif isinstance(comp, OpenVRMovementSink):
            comp._transport = RelayPoseTransport(relay_bridge)
        elif isinstance(comp, VRChatVideo):
            comp._transport = RelayVideoSourceTransport(relay_bridge)


def start_all(
    manager: GraphManager,
    ui_bridge: UIChannelBridge,
    relay_bridge: RelayBridge | None = None,
) -> None:
    if relay_bridge is not None and relay_bridge.is_connected:
        _inject_relay_transports(manager, relay_bridge)

    recv_overrides, send_overrides = ui_bridge.wire(manager)
    manager.run(recv_overrides, send_overrides)


def stop_all(manager: GraphManager) -> None:
    manager.stop()
