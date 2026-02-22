from __future__ import annotations

import time
from typing import NamedTuple

import cv2

from src.core.channel import Sender
from src.core.component import Component


class VideoPlayerOutputs(NamedTuple):
    video: Sender[bytes]


class VideoPlayer(Component[tuple[()], VideoPlayerOutputs]):
    """Plays a local video file and sends JPEG frames through the pipeline."""

    def __init__(
        self,
        *,
        path: str = "/Users/kevin/Downloads/1234.mov",
        quality: int = 75,
    ) -> None:
        super().__init__()
        self._path = path
        self._quality = quality

    def run(self, inputs: tuple[()], outputs: VideoPlayerOutputs) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps

        try:
            next_time = time.monotonic()

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                _, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
                )
                outputs.video.send(buf.tobytes())

                next_time += interval
                sleep = next_time - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_time = time.monotonic()
        finally:
            cap.release()
