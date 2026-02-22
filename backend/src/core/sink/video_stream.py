from __future__ import annotations

import threading
from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import Component


class VideoStreamInputs(NamedTuple):
    video: Receiver[bytes]


class VideoStream(Component[VideoStreamInputs, tuple[()]]):
    """Receives JPEG video frames and makes them available for frontend streaming."""

    def __init__(self) -> None:
        super().__init__()
        self._latest_frame: bytes | None = None
        self._frame_event = threading.Event()

    @property
    def latest_frame(self) -> bytes | None:
        return self._latest_frame

    def wait_for_frame(self, timeout: float = 1.0) -> bytes | None:
        """Block until a new frame arrives or timeout. Returns the frame or None."""
        self._frame_event.wait(timeout)
        self._frame_event.clear()
        return self._latest_frame

    def run(self, inputs: VideoStreamInputs, outputs: tuple[()]) -> None:
        for frame in inputs.video(self):
            if frame is None:
                break
            self._latest_frame = frame
            self._frame_event.set()
