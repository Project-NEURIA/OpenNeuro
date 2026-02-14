from __future__ import annotations

from pydantic import BaseModel


class FrameSnapshot(BaseModel):
    id: int
    frame_type_string: str
    pts: int
    size_bytes: int
    message: str  # __str__ representation


class FramesResponse(BaseModel):
    frames: list[FrameSnapshot]
    timestamp: float
