"""Relay bridge: manages a WebSocket connection to a local relay client.

The relay client runs on the user's machine alongside VRChat/SteamVR and
forwards OSC, pose, and video data between the remote backend and local
hardware.

Wire format (matches UIChannelBridge conventions):
    Binary frames: 2-byte header length (big-endian) + JSON header + raw payload
    Text frames:   JSON with a "ch" field for routing
"""

from __future__ import annotations

import asyncio
import json
import queue
import struct
import threading
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


class RelayBridge:
    def __init__(self) -> None:
        self._ws: WebSocket | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = threading.Event()

        # Video frames arriving from the relay → consumed by RelayVideoSourceTransport
        self._video_frame_queue: queue.Queue[tuple[dict, bytes]] = queue.Queue(
            maxsize=2
        )

        # Camera init data (intrinsics, IPD) — sent once when relay connects
        self._camera_init: dict[str, Any] | None = None
        self._camera_init_event = threading.Event()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # -- WebSocket lifecycle --

    async def run(self, ws: WebSocket) -> None:
        """Own the relay WebSocket lifecycle.  Blocks until disconnect."""
        self._ws = ws
        self._loop = asyncio.get_running_loop()
        self._connected.set()
        print("[relay] Client connected")
        try:
            await self._recv_loop(ws)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"[relay] Error in recv loop: {exc!r}")
        finally:
            self._connected.clear()
            self._ws = None
            self._loop = None
            # Unblock any waiting video consumers
            try:
                self._video_frame_queue.put_nowait(({"ch": "close"}, b""))
            except queue.Full:
                pass
            print("[relay] Client disconnected")

    # -- Thread-safe sends (called from component threads) --

    def send_json(self, msg: dict[str, Any]) -> None:
        ws, loop = self._ws, self._loop
        if ws is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(ws.send_json(msg), loop)

    def send_binary(self, header: dict[str, Any], payload: bytes) -> None:
        ws, loop = self._ws, self._loop
        if ws is not None and loop is not None:
            header_bytes = json.dumps(header).encode()
            data = struct.pack(">H", len(header_bytes)) + header_bytes + payload
            asyncio.run_coroutine_threadsafe(ws.send_bytes(data), loop)

    # -- Receive loop --

    async def _recv_loop(self, ws: WebSocket) -> None:
        while True:
            ws_msg = await ws.receive()
            msg_type = ws_msg.get("type")
            if msg_type == "websocket.disconnect":
                break
            if "bytes" in ws_msg:
                self._handle_binary(ws_msg["bytes"])
            elif "text" in ws_msg:
                self._handle_json(json.loads(ws_msg["text"]))

    def _handle_binary(self, data: bytes) -> None:
        if len(data) < 2:
            return
        header_len = struct.unpack(">H", data[:2])[0]
        header = json.loads(data[2 : 2 + header_len].decode())
        payload = data[2 + header_len :]

        ch = header.get("ch")
        if ch == "video":
            # Drop oldest if full (newest-only semantics for video)
            if self._video_frame_queue.full():
                try:
                    self._video_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._video_frame_queue.put_nowait((header, payload))

    def _handle_json(self, msg: dict[str, Any]) -> None:
        ch = msg.get("ch")
        if ch == "video_init":
            self._camera_init = msg
            self._camera_init_event.set()
