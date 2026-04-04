"""UI channel bridge: creates UI channels and manages the WebSocket connection.

The bridge inspects components for UI slots, creates channels and handles,
passes component-side handles as overrides to GraphManager.run(), and
owns the WebSocket lifecycle for bidirectional UI communication.

Wire format:
    Binary frames: 2-byte header length (big-endian) + JSON header + raw payload
    Text frames:   JSON {"type": "ui_input"|"ui_output", "node_id", "channel", "payload"}
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from typing import Any, get_args, get_origin

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from src.core.channel import Channel, Receiver, Sender, UISender, UIReceiver
from src.core.frames import TextFrame
from src.core.graph import GraphManager
from src.core.utils import ReceiverKey, SenderKey


# -- Wire format: encode/decode pairs --

def encode_binary(node_id: str, slot: str, payload: bytes) -> bytes:
    """Pack a binary UI message: 2-byte header len + JSON header + payload."""
    header = json.dumps({"type": "ui_output", "node_id": node_id, "channel": slot}).encode()
    return struct.pack(">H", len(header)) + header + payload


def decode_binary(buf: bytes) -> tuple[SenderKey | None, bytes]:
    """Unpack a binary UI message. Returns (key, payload)."""
    if len(buf) < 2:
        return None, b""
    header_len = struct.unpack(">H", buf[:2])[0]
    header = json.loads(buf[2 : 2 + header_len].decode("utf-8"))
    payload = buf[2 + header_len :]
    return (header.get("node_id"), header.get("channel")), payload


def encode_json(node_id: str, slot: str, item: Any) -> dict[str, Any]:
    """Build a JSON UI output envelope."""
    payload: Any
    if isinstance(item, BaseModel):
        payload = item.model_dump()
    elif isinstance(item, TextFrame):
        payload = item.text
    else:
        payload = item
    return {"type": "ui_output", "node_id": node_id, "channel": slot, "payload": payload}


def decode_json(text: str) -> tuple[SenderKey, Any] | None:
    """Parse a JSON UI input message. Returns (key, raw_payload) or None."""
    msg = json.loads(text)
    if msg.get("type") != "ui_input":
        return None
    return (msg["node_id"], msg["channel"]), msg.get("payload", "")


def deserialize_payload(payload: Any, inner_type: type | None) -> Any:
    """Convert a raw JSON payload to the expected type."""
    if inner_type is not None and issubclass(inner_type, BaseModel) and isinstance(payload, dict):
        return inner_type.model_validate(payload)
    if inner_type is not None and hasattr(inner_type, "new") and isinstance(payload, str):
        return inner_type.new(text=payload)
    return payload


# -- Bridge --

class UIChannelBridge:
    def __init__(self) -> None:
        self._ui_senders: dict[SenderKey, Sender[Any]] = {}
        self._ui_receivers: dict[ReceiverKey, Receiver[Any]] = {}
        self._manager: GraphManager | None = None
        self._ws: WebSocket | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._send_tasks: dict[SenderKey, asyncio.Task[None]] = {}

    def wire(
        self, manager: GraphManager
    ) -> tuple[
        dict[ReceiverKey, Receiver[Any] | None],
        dict[SenderKey, Sender[Any] | None],
    ]:
        """Create UI channels for all components and return overrides for run()."""
        self._manager = manager
        self._ui_senders.clear()
        self._ui_receivers.clear()

        recv_overrides: dict[ReceiverKey, Receiver[Any] | None] = {}
        send_overrides: dict[SenderKey, Sender[Any] | None] = {}

        for node_id, comp in manager.components().items():
            stop_event = (
                comp.stop_event
                if hasattr(comp, "stop_event")
                else threading.Event()
            )

            for slot, slot_type in comp.get_ui_input_types().items():
                ch: Channel[Any] = Channel()
                origin = get_origin(slot_type) or slot_type
                recv_overrides[(node_id, slot)] = origin(ch, stop_event)
                self._ui_senders[(node_id, slot)] = Sender(ch)

            for slot, slot_type in comp.get_ui_output_types().items():
                ch = Channel()
                origin = get_origin(slot_type) or slot_type
                send_overrides[(node_id, slot)] = origin(ch)
                self._ui_receivers[(node_id, slot)] = Receiver(ch, threading.Event())

        # Restart outbound tasks if WS is connected
        if self._ws is not None:
            self._send_msgs(self._ws)

        return recv_overrides, send_overrides

    async def run(self, ws: WebSocket) -> None:
        """Own the WebSocket: read inbound, send outbound via tasks."""
        self._ws = ws
        self._stop_event.clear()
        self._send_msgs(ws)
        try:
            await self._recv_msgs(ws)
        finally:
            self._ws = None
            self._stop_event.set()
            for t in self._send_tasks.values():
                t.cancel()
            await asyncio.gather(*self._send_tasks.values(), return_exceptions=True)
            self._send_tasks.clear()

    # -- Outbound: component → frontend --

    def _send_msgs(self, ws: WebSocket) -> None:
        for key in list(self._send_tasks):
            self._send_tasks.pop(key).cancel()
        for key, receiver in self._ui_receivers.items():
            node_id, slot = key
            inner_type = self._resolve_ui_output_type(node_id, slot)
            self._send_tasks[key] = asyncio.create_task(
                self._send_msg_task(ws, node_id, slot, receiver, inner_type)
            )

    async def _send_msg_task(
        self, ws: WebSocket, node_id: str, slot: str,
        receiver: Receiver[Any], inner_type: type | None,
    ) -> None:
        try:
            while not self._stop_event.is_set():
                item = await asyncio.to_thread(next, receiver)
                if item is None:
                    break
                if inner_type is not None and issubclass(inner_type, bytes):
                    await ws.send_bytes(encode_binary(node_id, slot, item))
                else:
                    await ws.send_json(encode_json(node_id, slot, item))
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass

    # -- Inbound: frontend → component --

    async def _recv_msgs(self, ws: WebSocket) -> None:
        while True:
            ws_msg = await ws.receive()
            if "bytes" in ws_msg:
                key, payload = decode_binary(ws_msg["bytes"])
                if key is not None:
                    sender = self._ui_senders.get(key)
                    if sender is not None:
                        sender.send(payload)
            elif "text" in ws_msg:
                result = decode_json(ws_msg["text"])
                if result is not None:
                    key, payload = result
                    sender = self._ui_senders.get(key)
                    if sender is not None:
                        inner_type = self._resolve_ui_input_type(*key)
                        sender.send(deserialize_payload(payload, inner_type))

    # -- Type resolution helpers --

    def _resolve_ui_output_type(self, node_id: str, slot: str) -> type | None:
        if self._manager is None:
            return None
        comp = self._manager.components().get(node_id)
        if comp is None:
            return None
        slot_type = comp.get_ui_output_types().get(slot)
        if slot_type is None:
            return None
        args = get_args(slot_type)
        if args:
            return args[0]
        origin = get_origin(slot_type) or slot_type
        if isinstance(origin, type):
            for base in getattr(origin, "__orig_bases__", ()):
                base_origin = get_origin(base)
                if base_origin is not None and issubclass(base_origin, UISender):
                    base_args = get_args(base)
                    if base_args:
                        return base_args[0]
        return None

    def _resolve_ui_input_type(self, node_id: str, slot: str) -> type | None:
        if self._manager is None:
            return None
        comp = self._manager.components().get(node_id)
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
