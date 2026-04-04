"""Single WebSocket that bridges all UI channels between components and frontend.

Inbound (text JSON):
    {"type": "ui_input", "node_id": "...", "channel": "...", "payload": "..."}
    -> finds the Sender for (node_id, channel) and calls .send()

Outbound:
    bytes channels: binary WS frame = 2-byte header length (big-endian) + JSON header + raw bytes
    BaseModel channels: JSON {"type": "ui_output", ..., "payload": {model dict}}
    Legacy (.get()) channels: JSON {"type": "ui_output", ..., "payload": "string"}
"""

from __future__ import annotations

import asyncio
import itertools
import json
import struct
import threading
from typing import Any, get_args, get_origin

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from src.core.channel import Receiver, UISender, UIReceiver
from src.core.graph import GraphManager

router = APIRouter(prefix="/ui")

_sub_id_counter = itertools.count()


def _resolve_ui_output_type(
    manager: GraphManager, node_id: str, slot: str
) -> type | None:
    """Extract the inner T from UISender[T] for the given output slot."""
    comp = manager.components().get(node_id)
    if comp is None:
        return None
    slot_type = comp.get_ui_output_types().get(slot)
    if slot_type is None:
        return None
    # slot_type is e.g. UISender[bytes], UIVideoSender (which is UISender[bytes]), etc.
    # Try get_args on the slot_type first (for generic aliases like UISender[MyModel])
    args = get_args(slot_type)
    if args:
        return args[0]
    # For concrete subclasses like UIVideoSender, walk __orig_bases__
    origin = get_origin(slot_type) or slot_type
    if isinstance(origin, type):
        for base in getattr(origin, "__orig_bases__", ()):
            base_origin = get_origin(base)
            if base_origin is not None and issubclass(base_origin, UISender):
                base_args = get_args(base)
                if base_args:
                    return base_args[0]
    return None


def _resolve_ui_input_type(
    manager: GraphManager, node_id: str, slot: str
) -> type | None:
    """Extract the inner T from UIReceiver[T] for the given input slot."""
    comp = manager.components().get(node_id)
    if comp is None:
        return None
    slot_type = comp.get_ui_input_types().get(slot)
    if slot_type is None:
        return None
    args = get_args(slot_type)
    if args:
        return args[0]
    origin = get_origin(slot_type) or slot_type
    if isinstance(origin, type):
        for base in getattr(origin, "__orig_bases__", ()):
            base_origin = get_origin(base)
            if base_origin is not None and issubclass(base_origin, UIReceiver):
                base_args = get_args(base)
                if base_args:
                    return base_args[0]
    return None


async def _read_ui_output(
    ws: WebSocket,
    node_id: str,
    slot: str,
    receiver: Receiver[Any],
    inner_type: type | None,
    stop_event: asyncio.Event,
) -> None:
    """Async task: reads from a blocking Receiver and sends to the WebSocket."""
    thread_stop = threading.Event()
    sub_id = next(_sub_id_counter)
    receiver._channel._register(sub_id)

    try:
        while not stop_event.is_set():
            item = await asyncio.to_thread(receiver._channel._get, sub_id, thread_stop)
            if item is None:
                break
            try:
                if inner_type is not None and issubclass(inner_type, bytes):
                    # Binary path (video frames, etc.)
                    header = json.dumps(
                        {"type": "ui_output", "node_id": node_id, "channel": slot}
                    ).encode()
                    prefix = struct.pack(">H", len(header))
                    await ws.send_bytes(prefix + header + item)
                elif isinstance(item, BaseModel):
                    # Pydantic model → JSON dict payload
                    await ws.send_json(
                        {
                            "type": "ui_output",
                            "node_id": node_id,
                            "channel": slot,
                            "payload": item.model_dump(),
                        }
                    )
                elif hasattr(item, "get"):
                    # Legacy TextFrame-style with .get()
                    await ws.send_json(
                        {
                            "type": "ui_output",
                            "node_id": node_id,
                            "channel": slot,
                            "payload": item.get(),
                        }
                    )
                else:
                    await ws.send_json(
                        {
                            "type": "ui_output",
                            "node_id": node_id,
                            "channel": slot,
                            "payload": item,
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
                inner_type = _resolve_ui_output_type(manager, node_id, slot)
                tasks[key] = asyncio.create_task(
                    _read_ui_output(ws, node_id, slot, receiver, inner_type, stop_event)
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
    print("[ui_ws] Got connection request from:", ws.client)
    manager: GraphManager = ws.app.state.manager
    await ws.accept()
    print("[ui_ws] Accepted connection!!!")

    stop_event = asyncio.Event()
    tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    watcher = asyncio.create_task(_watch_ui_channels(ws, manager, stop_event, tasks))

    try:
        while True:
            ws_msg = await ws.receive()
            if "bytes" in ws_msg:
                # Binary frame: 2-byte header length + JSON header + payload
                buf = ws_msg["bytes"]
                if len(buf) >= 2:
                    header_len = struct.unpack(">H", buf[:2])[0]
                    header = json.loads(buf[2 : 2 + header_len].decode("utf-8"))
                    payload = buf[2 + header_len :]

                    node_id = header.get("node_id")
                    channel = header.get("channel")
                    sender = manager.ui_senders().get((node_id, channel))
                    if sender is not None:
                        sender.send(payload)
            elif "text" in ws_msg:
                msg = json.loads(ws_msg["text"])
                if msg.get("type") == "ui_input":
                    node_id = msg["node_id"]
                    channel = msg["channel"]
                    payload = msg.get("payload", "")
                    sender = manager.ui_senders().get((node_id, channel))
                    if sender is not None:
                        inner_type = _resolve_ui_input_type(manager, node_id, channel)
                        if (
                            inner_type is not None
                            and issubclass(inner_type, BaseModel)
                            and isinstance(payload, dict)
                        ):
                            sender.send(inner_type.model_validate(payload))
                        elif (
                            inner_type is not None
                            and hasattr(inner_type, "new")
                            and isinstance(payload, str)
                        ):
                            sender.send(inner_type.new(text=payload))
                        else:
                            sender.send(payload)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop_event.set()
        watcher.cancel()
        for t in tasks.values():
            t.cancel()
        await asyncio.gather(watcher, *tasks.values(), return_exceptions=True)
