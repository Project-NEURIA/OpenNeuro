#!/usr/bin/env python3
"""OpenNeuro Local Relay — bridges a remote backend to local VRChat/SteamVR.

Connects to the backend's /relay/ws WebSocket endpoint and forwards:
  - OSC messages (backend → local VRChat via UDP)
  - Body poses  (backend → local OpenVR driver via TCP)
  - Stereo video (local OpenVR driver → backend, optional)

Usage:
    python scripts/local_relay.py --backend ws://BACKEND_HOST:8000/relay/ws
    python scripts/local_relay.py --backend ws://BACKEND_HOST:8000/relay/ws --enable-video
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import struct
from typing import Any

import numpy as np
import websockets
from PIL import Image
from pythonosc.udp_client import SimpleUDPClient


def _encode_binary(header: dict[str, Any], payload: bytes) -> bytes:
    """Encode a binary relay message: 2-byte header_len + JSON header + payload."""
    header_bytes = json.dumps(header).encode()
    return struct.pack(">H", len(header_bytes)) + header_bytes + payload


def _decode_binary(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Decode a binary relay message."""
    header_len = struct.unpack(">H", data[:2])[0]
    header = json.loads(data[2 : 2 + header_len].decode())
    payload = data[2 + header_len :]
    return header, payload


def _compress_rgba(raw: bytes, w: int, h: int, quality: int = 80) -> bytes:
    """RGBA raw bytes → JPEG bytes."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
    rgb = arr[:, :, :3]
    img = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class LocalRelay:
    def __init__(
        self,
        backend_url: str,
        osc_host: str,
        osc_port: int,
        ovr_host: str,
        ovr_port: int,
        enable_video: bool,
        jpeg_quality: int,
    ) -> None:
        self.backend_url = backend_url
        self.osc = SimpleUDPClient(osc_host, osc_port)
        self.ovr_host = ovr_host
        self.ovr_port = ovr_port
        self.enable_video = enable_video
        self.jpeg_quality = jpeg_quality
        self._ovr_client: Any = None
        self._ws: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        print(f"[relay] Connecting to {self.backend_url} ...")
        async with websockets.connect(self.backend_url) as ws:
            self._ws = ws
            self._loop = asyncio.get_running_loop()
            print("[relay] Connected")
            tasks: list[asyncio.Task[Any]] = [
                asyncio.create_task(self._recv_loop(ws))
            ]
            if self.enable_video:
                tasks.append(
                    asyncio.create_task(asyncio.to_thread(self._video_capture_loop))
                )
            try:
                await asyncio.gather(*tasks)
            except websockets.exceptions.ConnectionClosed as exc:
                print(f"[relay] Connection closed: code={exc.code} reason={exc.reason}")
            except Exception as exc:
                print(f"[relay] Error: {exc!r}")

    # -- Receive from backend, forward to local apps --

    async def _recv_loop(self, ws: Any) -> None:
        async for message in ws:
            if isinstance(message, str):
                msg = json.loads(message)
                ch = msg.get("ch")
                if ch == "osc":
                    self.osc.send_message(msg["address"], msg["value"])
            elif isinstance(message, bytes):
                header, payload = _decode_binary(message)
                ch = header.get("ch")
                if ch == "pose":
                    self._forward_pose(payload)

    # -- Pose forwarding --

    BONE_ORDER = [
        "head",
        "left_hand",
        "right_hand",
        "waist",
        "chest",
        "left_foot",
        "right_foot",
        "left_knee",
        "right_knee",
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
    ]

    def _forward_pose(self, packed: bytes) -> None:
        try:
            from ovd_client import Client, Pose
        except ImportError:
            return

        if self._ovr_client is None:
            self._ovr_client = Client(host=self.ovr_host, port=self.ovr_port)
            self._ovr_client.connect()
            print(
                f"[relay] Connected to OpenVR driver at {self.ovr_host}:{self.ovr_port}"
            )

        poses: dict[str, Pose] = {}
        for i, name in enumerate(self.BONE_ORDER):
            offset = i * 28  # 7 floats × 4 bytes
            vals = struct.unpack("<7f", packed[offset : offset + 28])
            poses[name] = Pose(
                pos_x=vals[0],
                pos_y=vals[1],
                pos_z=vals[2],
                rot_w=vals[3],
                rot_x=vals[4],
                rot_y=vals[5],
                rot_z=vals[6],
            )
        self._ovr_client.update_pose(**poses)

    # -- Video capture (local → backend) --

    def _video_capture_loop(self) -> None:
        try:
            from ovd_client import Client
        except ImportError:
            print("[relay] ovd-client not installed — video capture disabled")
            return

        with Client(host=self.ovr_host, port=self.ovr_port) as client:
            intrinsics = client.get_intrinsics()
            ipd = client.get_ipd()

            init_msg = json.dumps(
                {
                    "ch": "video_init",
                    "ipd": float(ipd),
                    "intrinsics": intrinsics.flatten().tolist(),
                    "w": 0,
                    "h": 0,  # filled on first frame
                }
            )
            self._send_async(init_msg)

            with client.frame_stream() as frames:
                for frame in frames:
                    left_jpeg = _compress_rgba(
                        frame.left, frame.width, frame.height, self.jpeg_quality
                    )
                    right_jpeg = _compress_rgba(
                        frame.right, frame.width, frame.height, self.jpeg_quality
                    )

                    ext = client.get_extrinsics(frame, eye=0)

                    header = {
                        "ch": "video",
                        "w": frame.width,
                        "h": frame.height,
                        "left_len": len(left_jpeg),
                        "extrinsics": ext.flatten().tolist(),
                    }
                    msg = _encode_binary(header, left_jpeg + right_jpeg)
                    self._send_async(msg)

    def _send_async(self, data: str | bytes) -> None:
        ws, loop = self._ws, self._loop
        if ws is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(ws.send(data), loop)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenNeuro Local Relay — bridges remote backend to local VRChat/SteamVR"
    )
    parser.add_argument(
        "--backend",
        required=True,
        help="WebSocket URL, e.g. ws://localhost:8000/relay/ws",
    )
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9000)
    parser.add_argument("--ovr-host", default="127.0.0.1")
    parser.add_argument("--ovr-port", type=int, default=21213)
    parser.add_argument(
        "--enable-video",
        action="store_true",
        help="Capture stereo video from SteamVR and stream to backend",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for video compression (1-100, default 80)",
    )
    args = parser.parse_args()

    relay = LocalRelay(
        backend_url=args.backend,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        ovr_host=args.ovr_host,
        ovr_port=args.ovr_port,
        enable_video=args.enable_video,
        jpeg_quality=args.jpeg_quality,
    )
    try:
        asyncio.run(relay.run())
    except KeyboardInterrupt:
        print("\n[relay] Shutting down")


if __name__ == "__main__":
    main()
