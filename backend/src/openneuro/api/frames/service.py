from __future__ import annotations

import time

from openneuro.core.frames import get_frame_registry
from openneuro.api.frames.dto.frames import FrameSnapshot, FramesResponse


def collect() -> FramesResponse:
    """Collect current frame snapshots from registry."""
    registry = get_frame_registry()
    frames = registry.get_all()

    snapshots = [
        FrameSnapshot(
            id=frame.id,
            frame_type_string=frame.frame_type_string,
            pts=frame.pts,
            size_bytes=frame.size_bytes(),
            message=str(frame),  # Use __str__ representation
        )
        for frame in frames
    ]

    return FramesResponse(
        frames=snapshots,
        timestamp=time.time(),
    )
