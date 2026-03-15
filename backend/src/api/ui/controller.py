"""Single WebSocket that bridges all UI channels between components and frontend.

Inbound (text JSON):
    {"type": "ui_input", "node_id": "...", "channel": "...", "payload": "..."}
    -> finds the Sender for (node_id, channel) and calls .send()

Outbound:
    Text channels: JSON {"type": "ui_output", "node_id": "...", "channel": "...", "payload": "..."}
    Video channels: binary WS frame = 2-byte header length (big-endian) + JSON header + raw JPEG bytes
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.core.channel import Receiver
from src.core.frames import TextFrame
from src.core.graph import GraphManager
from src.core.channel import UIVideoSender

router = APIRouter(prefix="/ui")


def _is_video_channel(manager: GraphManager, node_id: str, slot: str) -> bool:
    """Check if the UI output slot is a UIVideoSender."""
    comp = manager.components().get(node_id)
    if comp is None:
        return False
    ui_outputs = type(comp).get_ui_output_types()
    slot_type = ui_outputs.get(slot)
    if slot_type is None:
        return False
    from typing import get_origin

    origin = get_origin(slot_type) or slot_type
    return isinstance(origin, type) and issubclass(origin, UIVideoSender)


async def _read_ui_output(
    ws: WebSocket,
    node_id: str,
    slot: str,
    receiver: Receiver[Any],
    is_video: bool,
    stop_event: asyncio.Event,
) -> None:
    """Async task: reads from a blocking Receiver and sends to the WebSocket."""
    thread_stop = threading.Event()
    sub_id = id(object())
    receiver._channel._register(sub_id)

    try:
        while not stop_event.is_set():
            item = await asyncio.to_thread(
                receiver._channel._wait_and_get, sub_id, thread_stop
            )
            if item is None:
                break
            try:
                if is_video:
                    header = json.dumps(
                        {"type": "ui_output", "node_id": node_id, "channel": slot}
                    ).encode()
                    prefix = struct.pack(">H", len(header))
                    await ws.send_bytes(prefix + header + item)
                else:
                    payload = item.get() if hasattr(item, "get") else str(item)
                    await ws.send_json(
                        {
                            "type": "ui_output",
                            "node_id": node_id,
                            "channel": slot,
                            "payload": payload,
                        }
                    )
            except (WebSocketDisconnect, RuntimeError):
                break
    finally:
        receiver._channel._unregister(sub_id)
        thread_stop.set()


async def _watch_ui_channels(
    ws: WebSocket,
    manager: GraphManager,
    stop_event: asyncio.Event,
    tasks: dict[tuple[str, str], asyncio.Task[None]],
) -> None:
    """Wait for GraphManager.run() to signal new UI channels, then spawn readers."""
    last_version = manager._ui_version
    while not stop_event.is_set():
        # Spawn tasks for any receivers we haven't seen yet
        for key, receiver in manager.ui_receivers().items():
            if key not in tasks:
                node_id, slot = key
                is_video = _is_video_channel(manager, node_id, slot)
                tasks[key] = asyncio.create_task(
                    _read_ui_output(ws, node_id, slot, receiver, is_video, stop_event)
                )

        if manager._ui_version == last_version:
            # Wait until run() fires the event
            try:
                await manager._ui_changed.wait()
            except asyncio.CancelledError:
                return

        # Cancel all stale tasks — even if the same keys exist, the
        # underlying Receiver objects point to new channels after a restart.
        for key in list(tasks):
            tasks.pop(key).cancel()

        last_version = manager._ui_version


@router.websocket("/ws")
async def ui_ws(ws: WebSocket) -> None:
    manager: GraphManager = ws.app.state.manager
    await ws.accept()

    stop_event = asyncio.Event()
    tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    watcher = asyncio.create_task(_watch_ui_channels(ws, manager, stop_event, tasks))

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ui_input":
                node_id = msg["node_id"]
                channel = msg["channel"]
                payload = msg.get("payload", "")
                sender = manager.ui_senders().get((node_id, channel))
                if sender is not None:
                    sender.send(TextFrame.new(text=payload))
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        watcher.cancel()
        for t in tasks.values():
            t.cancel()
        await asyncio.gather(watcher, *tasks.values(), return_exceptions=True)
