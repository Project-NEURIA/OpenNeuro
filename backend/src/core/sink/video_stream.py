from __future__ import annotations

import threading
from typing import TypedDict

from src.core.component import Component
from src.core.channel import Channel


class VideoStreamOutputs(TypedDict):
    pass


class VideoStream(Component[[Channel[bytes]], VideoStreamOutputs]):
    """Receives JPEG video frames and makes them available for frontend streaming."""

    def __init__(self) -> None:
        super().__init__()
        self._latest_frame: bytes | None = None
        self._frame_event = threading.Event()

    def get_output_channels(self) -> VideoStreamOutputs:
        return {}

    @property
    def latest_frame(self) -> bytes | None:
        return self._latest_frame

    def wait_for_frame(self, timeout: float = 1.0) -> bytes | None:
        """Block until a new frame arrives or timeout. Returns the frame or None."""
        self._frame_event.wait(timeout)
        self._frame_event.clear()
        return self._latest_frame

    def run(self, video: Channel[bytes]) -> None:
        for frame in video.stream(self):
            if frame is None:
                break
            self._latest_frame = frame
            self._frame_event.set()
