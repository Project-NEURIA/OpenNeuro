"""Transport abstractions for OSC and OpenVR protocols.

Each protocol has a base class with local (direct UDP/TCP) and relay
(WebSocket-mediated) implementations.  Components receive one of these
at construction time; the transport used depends on whether a relay
client is connected.
"""

from __future__ import annotations

import io
import json
import queue
import struct
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

if TYPE_CHECKING:
    from src.api.relay.bridge import RelayBridge


# ── OSC ────────────────────────────────────────────────────────────


class OscTransport(ABC):
    """Abstract OSC message sender."""

    @abstractmethod
    def send_message(self, address: str, value: Any) -> None: ...

    def close(self) -> None:
        pass


class LocalOscTransport(OscTransport):
    """Sends OSC messages directly via UDP."""

    def __init__(self, host: str, port: int) -> None:
        from pythonosc.udp_client import SimpleUDPClient

        self._client = SimpleUDPClient(host, port)
        self._lock = threading.Lock()

    def send_message(self, address: str, value: Any) -> None:
        with self._lock:
            self._client.send_message(address, value)


# ── OpenVR Poses ───────────────────────────────────────────────────


class PoseTransport(ABC):
    """Abstract body-pose sender (wraps ovd_client.Client for output)."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def update_pose(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...


class LocalPoseTransport(PoseTransport):
    """Sends poses directly via TCP to the OpenVR virtual driver."""

    def __init__(self, host: str, port: int) -> None:
        from ovd_client import Client

        self._client = Client(host=host, port=port)

    def connect(self) -> None:
        self._client.connect()

    def update_pose(self, **kwargs: Any) -> None:
        self._client.update_pose(**kwargs)

    def disconnect(self) -> None:
        self._client.disconnect()


# ── OpenVR Video ───────────────────────────────────────────────────


@dataclass
class StereoFrame:
    """Minimal frame object compatible with ovd_client frame objects."""

    left: bytes
    right: bytes
    width: int
    height: int


class VideoSourceTransport(ABC):
    """Abstract stereo video source (wraps ovd_client.Client for input)."""

    @abstractmethod
    def get_intrinsics(self) -> np.ndarray: ...

    @abstractmethod
    def get_ipd(self) -> float: ...

    @abstractmethod
    def get_extrinsics(self, frame: Any, eye: int = 0) -> np.ndarray: ...

    @abstractmethod
    @contextmanager
    def frame_stream(self) -> Generator[Any, None, None]: ...

    def __enter__(self) -> VideoSourceTransport:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class LocalVideoSourceTransport(VideoSourceTransport):
    """Captures stereo video directly from the OpenVR virtual driver."""

    def __init__(self, host: str, port: int) -> None:
        from ovd_client import Client

        self._client = Client(host=host, port=port)

    def __enter__(self) -> LocalVideoSourceTransport:
        self._client.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.__exit__(*args)

    def get_intrinsics(self) -> np.ndarray:
        return self._client.get_intrinsics()

    def get_ipd(self) -> float:
        return self._client.get_ipd()

    def get_extrinsics(self, frame: Any, eye: int = 0) -> np.ndarray:
        return self._client.get_extrinsics(frame, eye=eye)

    @contextmanager
    def frame_stream(self) -> Generator[Any, None, None]:
        with self._client.frame_stream() as frames:
            yield frames


# ── Relay Implementations ──────────────────────────────────────────

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


class RelayOscTransport(OscTransport):
    """Sends OSC messages via the relay WebSocket."""

    def __init__(self, bridge: RelayBridge) -> None:
        self._bridge = bridge

    def send_message(self, address: str, value: Any) -> None:
        self._bridge.send_json(
            {
                "ch": "osc",
                "address": address,
                "value": value if isinstance(value, list) else [value],
            }
        )


class RelayPoseTransport(PoseTransport):
    """Sends poses as packed binary via the relay WebSocket."""

    def __init__(self, bridge: RelayBridge) -> None:
        self._bridge = bridge

    def connect(self) -> None:
        pass

    def update_pose(self, **kwargs: Any) -> None:
        # Pack each bone as 7 little-endian floats (pos_xyz + rot_wxyz)
        parts: list[bytes] = []
        for name in BONE_ORDER:
            pose = kwargs.get(name)
            if pose is None:
                parts.append(struct.pack("<7f", 0, 0, 0, 1, 0, 0, 0))
            else:
                parts.append(
                    struct.pack(
                        "<7f",
                        pose.pos_x,
                        pose.pos_y,
                        pose.pos_z,
                        pose.rot_w,
                        pose.rot_x,
                        pose.rot_y,
                        pose.rot_z,
                    )
                )
        self._bridge.send_binary({"ch": "pose"}, b"".join(parts))

    def disconnect(self) -> None:
        pass


class RelayVideoSourceTransport(VideoSourceTransport):
    """Receives JPEG-compressed stereo frames from the relay WebSocket."""

    def __init__(self, bridge: RelayBridge) -> None:
        self._bridge = bridge

    def get_intrinsics(self) -> np.ndarray:
        self._bridge._camera_init_event.wait(timeout=30)
        init = self._bridge._camera_init
        if init is None:
            raise RuntimeError("Relay did not send camera init")
        return np.array(init["intrinsics"], dtype=np.float64).reshape(3, 3)

    def get_ipd(self) -> float:
        self._bridge._camera_init_event.wait(timeout=30)
        init = self._bridge._camera_init
        if init is None:
            raise RuntimeError("Relay did not send camera init")
        return float(init["ipd"])

    def get_extrinsics(self, frame: Any, eye: int = 0) -> np.ndarray:
        # The relay sends extrinsics embedded in each frame header
        if hasattr(frame, "extrinsics") and frame.extrinsics is not None:
            return frame.extrinsics
        return np.eye(4, dtype=np.float64)

    @contextmanager
    def frame_stream(self) -> Generator[Any, None, None]:
        yield self._frame_iter()

    def _frame_iter(self) -> Generator[Any, None, None]:
        from PIL import Image

        while True:
            try:
                header, payload = self._bridge._video_frame_queue.get(timeout=0.5)
            except queue.Empty:
                if not self._bridge.is_connected:
                    break
                continue
            if header.get("ch") == "close":
                break

            w = header["w"]
            h = header["h"]
            left_len = header["left_len"]
            left_jpeg = payload[:left_len]
            right_jpeg = payload[left_len:]

            # Decode JPEG → raw RGBA bytes for compatibility with ovd_client frames
            left_rgb = np.array(Image.open(io.BytesIO(left_jpeg)).convert("RGBA"))
            right_rgb = np.array(Image.open(io.BytesIO(right_jpeg)).convert("RGBA"))

            frame = StereoFrame(
                left=left_rgb.tobytes(),
                right=right_rgb.tobytes(),
                width=w,
                height=h,
            )

            # Attach extrinsics from header if present
            if "extrinsics" in header:
                frame.extrinsics = np.array(  # type: ignore[attr-defined]
                    header["extrinsics"], dtype=np.float64
                ).reshape(4, 4)

            yield frame
