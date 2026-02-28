from __future__ import annotations

import time
from abc import abstractmethod
from typing import NamedTuple

import cv2
import numpy as np
from pydantic import BaseModel

from src.core.channel import Sender
from src.core.component import Component
from src.core.frames import VideoFrame


class VideoSourceConfig(BaseModel):
    source: str = ""
    width_resize: int | None = None
    height_resize: int | None = None
    fps_resample: int | None = None


class VideoSourceOutputs(NamedTuple):
    video: Sender[VideoFrame]


class VideoSource(Component[tuple[()], VideoSourceOutputs]):
    """Abstract base for components that read video via cv2.VideoCapture."""

    @abstractmethod
    def __init__(self, config: VideoSourceConfig) -> None:
        super().__init__()
        self._config = config

    def _open_capture(self) -> cv2.VideoCapture:
        source: int | str
        try:
            source = int(self._config.source)
        except ValueError:
            source = self._config.source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._config.source}")
        return cap

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        cfg_w = self._config.width_resize
        cfg_h = self._config.height_resize
        if cfg_w is None and cfg_h is None:
            return frame

        h, w = frame.shape[:2]

        if cfg_w is not None and cfg_h is not None:
            # Both set: resize to fill target box, then center-crop.
            scale = max(cfg_w / w, cfg_h / h)
            sw, sh = round(w * scale), round(h * scale)
            resized = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
            x0, y0 = (sw - cfg_w) // 2, (sh - cfg_h) // 2
            return resized[y0 : y0 + cfg_h, x0 : x0 + cfg_w]  # type: ignore[return-value]
        elif cfg_w is not None:
            target_h = round(h * (cfg_w / w))
            return cv2.resize(frame, (cfg_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            assert cfg_h is not None
            target_w = round(w * (cfg_h / h))
            return cv2.resize(frame, (target_w, cfg_h), interpolation=cv2.INTER_LINEAR)

    def _on_eof(self, cap: cv2.VideoCapture) -> None:
        """Called when cap.read() returns False. Override to customize EOF behavior."""

    def run(self, inputs: tuple[()], outputs: VideoSourceOutputs) -> None:
        cap = self._open_capture()

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        output_fps = self._config.fps_resample or native_fps
        read_interval = 1.0 / native_fps
        send_interval = 1.0 / output_fps

        try:
            next_read_time = time.monotonic()
            last_send_time = 0.0

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self._on_eof(cap)
                    continue

                now = time.monotonic()
                if now - last_send_time >= send_interval:
                    outputs.video.send(VideoFrame.new(data=self._resize(frame)))
                    last_send_time = now

                next_read_time += read_interval
                sleep = next_read_time - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_read_time = time.monotonic()
        finally:
            cap.release()
