from __future__ import annotations

import time
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict

from src.core.channel import Sender
from src.core.component import Component
from src.core.frames import VideoFrame


class CameraConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"configOptions": {"source": {}}})

    source: str = "0"
    width: int | None = None
    height: int | None = None
    fps: int | None = None


class CameraOutputs(NamedTuple):
    video: Sender[VideoFrame]


class Camera(Component[tuple[()], CameraOutputs]):
    """Captures live video from a webcam or camera device."""

    def __init__(self, config: CameraConfig = CameraConfig()) -> None:
        super().__init__()
        self._config = config

    def run(self, inputs: tuple[()], outputs: CameraOutputs) -> None:
        import cv2

        source: int | str
        try:
            source = int(self._config.source)
        except ValueError:
            source = self._config.source

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self._config.source}")

        if self._config.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        if self._config.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        if self._config.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps

        try:
            next_time = time.monotonic()

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue

                outputs.video.send(VideoFrame.new(data=frame))

                next_time += interval
                sleep = next_time - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_time = time.monotonic()
        finally:
            cap.release()

    @classmethod
    def get_config_options(
        cls, field: str, values: dict[str, Any] | None = None
    ) -> list[dict[str, str]] | None:
        if field != "config.source":
            return None

        results: list[dict[str, str]] = []
        try:
            from cv2_enumerate_cameras import enumerate_cameras

            for cam in enumerate_cameras():
                idx = str(cam.index)
                label = cam.name if cam.name else f"Camera {idx}"
                results.append({"value": idx, "label": label})
        except Exception:
            # Fallback: probe first few indices
            import cv2

            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    results.append({"value": str(i), "label": f"Camera {i}"})
                    cap.release()

        return results
