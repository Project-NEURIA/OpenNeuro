"""
Binary WebSocket protocol for FFS stereo depth + nvblox TSDF fusion.

Lightweight module (only stdlib + numpy) so it can be shared between
the server (this repo) and the OpenNeuro client component.

Message layout
--------------
Every message starts with a 1-byte **tag** followed by a type-specific
binary payload.  Control messages (DECAY, CLEAR, CLOSE, *_ACK) are just
the tag byte with no payload.

Client → Server:
    0x01  INIT          JSON config (UTF-8)
    0x02  FRAME         [4B left_len][4B right_len][64B pose f32x16][left_jpg][right_jpg]
    0x03  DECAY         (no payload)
    0x04  CLEAR         (no payload)
    0x05  CLOSE         (no payload)

Server → Client:
    0x81  INIT_ACK      JSON status (UTF-8)
    0x82  FRAME_RESULT  [4B flags][2B H][2B W][depth/stereo data …]
    0x83  DECAY_ACK     (no payload)
    0x84  CLEAR_ACK     (no payload)
    0x85  ERROR         UTF-8 error string

FRAME_RESULT flags (uint32 LE bitfield):
    bit 0  has_d_map      → H*W*2 bytes float16
    bit 1  has_d_live     → H*W*2 bytes float16
    bit 2  has_stereo     → [4B left_len][4B right_len][left_jpg][right_jpg]
    bit 3  vram_warning   → server VRAM above threshold
"""

from __future__ import annotations

import json
import struct
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Tag constants
# ---------------------------------------------------------------------------

# Client → Server
MSG_INIT = 0x01
MSG_FRAME = 0x02
MSG_DECAY = 0x03
MSG_CLEAR = 0x04
MSG_CLOSE = 0x05

# Server → Client
MSG_INIT_ACK = 0x81
MSG_FRAME_RESULT = 0x82
MSG_DECAY_ACK = 0x83
MSG_CLEAR_ACK = 0x84
MSG_ERROR = 0x85

# FRAME_RESULT flag bits
FLAG_HAS_D_MAP = 1 << 0
FLAG_HAS_D_LIVE = 1 << 1
FLAG_HAS_STEREO = 1 << 2
FLAG_VRAM_WARNING = 1 << 3

# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

def pack_init(config: dict) -> bytes:
    payload = json.dumps(config).encode("utf-8")
    return bytes([MSG_INIT]) + payload


def unpack_init(data: bytes) -> dict:
    assert data[0] == MSG_INIT
    return json.loads(data[1:].decode("utf-8"))

# ---------------------------------------------------------------------------
# FRAME  (client → server)
# ---------------------------------------------------------------------------

_FRAME_HEADER = struct.Struct("<II")  # left_len, right_len
_POSE_SIZE = 64  # 16 x float32 = 64 bytes


def pack_frame(left_jpeg: bytes, right_jpeg: bytes,
               pose: np.ndarray) -> bytes:
    """Pack a stereo frame + 4x4 pose into a binary message."""
    pose_bytes = pose.astype(np.float32).tobytes()
    assert len(pose_bytes) == _POSE_SIZE
    header = _FRAME_HEADER.pack(len(left_jpeg), len(right_jpeg))
    return bytes([MSG_FRAME]) + header + pose_bytes + left_jpeg + right_jpeg


def unpack_frame(data: bytes) -> tuple[bytes, bytes, np.ndarray]:
    """Unpack → (left_jpeg, right_jpeg, pose_4x4)."""
    assert data[0] == MSG_FRAME
    off = 1
    left_len, right_len = _FRAME_HEADER.unpack_from(data, off)
    off += _FRAME_HEADER.size
    pose = np.frombuffer(data, dtype=np.float32, count=16,
                         offset=off).reshape(4, 4).copy()
    off += _POSE_SIZE
    left_jpeg = data[off:off + left_len]
    off += left_len
    right_jpeg = data[off:off + right_len]
    return left_jpeg, right_jpeg, pose

# ---------------------------------------------------------------------------
# FRAME_RESULT  (server → client)
# ---------------------------------------------------------------------------

_RESULT_HEADER = struct.Struct("<IHH")  # flags, H, W
_JPEG_LEN_PAIR = struct.Struct("<II")   # left_len, right_len


class FrameResult(NamedTuple):
    d_map: np.ndarray | None       # (H, W) float32
    d_live: np.ndarray | None      # (H, W) float32
    left_jpeg: bytes | None
    right_jpeg: bytes | None
    height: int
    width: int
    vram_warning: bool


def pack_frame_result(
    *,
    height: int,
    width: int,
    d_map: np.ndarray | None = None,
    d_live: np.ndarray | None = None,
    left_jpeg: bytes | None = None,
    right_jpeg: bytes | None = None,
    vram_warning: bool = False,
) -> bytes:
    flags = 0
    if d_map is not None:
        flags |= FLAG_HAS_D_MAP
    if d_live is not None:
        flags |= FLAG_HAS_D_LIVE
    if left_jpeg is not None and right_jpeg is not None:
        flags |= FLAG_HAS_STEREO
    if vram_warning:
        flags |= FLAG_VRAM_WARNING

    parts = [
        bytes([MSG_FRAME_RESULT]),
        _RESULT_HEADER.pack(flags, height, width),
    ]

    if d_map is not None:
        parts.append(d_map.astype(np.float16).tobytes())
    if d_live is not None:
        parts.append(d_live.astype(np.float16).tobytes())
    if flags & FLAG_HAS_STEREO:
        parts.append(_JPEG_LEN_PAIR.pack(len(left_jpeg), len(right_jpeg)))
        parts.append(left_jpeg)
        parts.append(right_jpeg)

    return b"".join(parts)


def unpack_frame_result(data: bytes) -> FrameResult:
    assert data[0] == MSG_FRAME_RESULT
    off = 1
    flags, H, W = _RESULT_HEADER.unpack_from(data, off)
    off += _RESULT_HEADER.size

    npix = H * W

    d_map = None
    if flags & FLAG_HAS_D_MAP:
        d_map = np.frombuffer(data, dtype=np.float16, count=npix,
                              offset=off).reshape(H, W).astype(np.float32)
        off += npix * 2

    d_live = None
    if flags & FLAG_HAS_D_LIVE:
        d_live = np.frombuffer(data, dtype=np.float16, count=npix,
                               offset=off).reshape(H, W).astype(np.float32)
        off += npix * 2

    left_jpeg = right_jpeg = None
    if flags & FLAG_HAS_STEREO:
        left_len, right_len = _JPEG_LEN_PAIR.unpack_from(data, off)
        off += _JPEG_LEN_PAIR.size
        left_jpeg = data[off:off + left_len]
        off += left_len
        right_jpeg = data[off:off + right_len]

    return FrameResult(
        d_map=d_map, d_live=d_live,
        left_jpeg=left_jpeg, right_jpeg=right_jpeg,
        height=H, width=W,
        vram_warning=bool(flags & FLAG_VRAM_WARNING),
    )

# ---------------------------------------------------------------------------
# Simple control messages (tag-only or tag + payload)
# ---------------------------------------------------------------------------

def pack_decay() -> bytes:
    return bytes([MSG_DECAY])

def pack_clear() -> bytes:
    return bytes([MSG_CLEAR])

def pack_close() -> bytes:
    return bytes([MSG_CLOSE])

def pack_init_ack(info: dict) -> bytes:
    return bytes([MSG_INIT_ACK]) + json.dumps(info).encode("utf-8")

def unpack_init_ack(data: bytes) -> dict:
    assert data[0] == MSG_INIT_ACK
    return json.loads(data[1:].decode("utf-8"))

def pack_decay_ack() -> bytes:
    return bytes([MSG_DECAY_ACK])

def pack_clear_ack() -> bytes:
    return bytes([MSG_CLEAR_ACK])

def pack_error(message: str) -> bytes:
    return bytes([MSG_ERROR]) + message.encode("utf-8")

def unpack_error(data: bytes) -> str:
    assert data[0] == MSG_ERROR
    return data[1:].decode("utf-8")

def msg_tag(data: bytes) -> int:
    """Return the message type tag (first byte)."""
    return data[0]
