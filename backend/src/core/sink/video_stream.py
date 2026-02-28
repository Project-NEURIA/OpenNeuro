from __future__ import annotations

import threading
from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import Component
from src.core.frames import VideoDataFormat, VideoFrame


class VideoStreamInputs(NamedTuple):
    video: Receiver[VideoFrame]


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
        import cv2

        for frame in inputs.video(self, newest=True):
            if frame is None:
                break
            _, buf = cv2.imencode(
                ".jpg",
                frame.get(VideoDataFormat.BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            self._latest_frame = buf.tobytes()
            self._frame_event.set()
