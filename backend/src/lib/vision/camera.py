from __future__ import annotations

from typing import Any

import cv2
from pydantic import ConfigDict

from src.lib.vision.video_source import VideoSource, VideoSourceConfig


class CameraConfig(VideoSourceConfig):
    model_config = ConfigDict(json_schema_extra={"options": {"source": {}}})
    # For the frontend to know which fields are dynamic

    source: str = "0"


class Camera(VideoSource):
    description = "Captures live **video** from a webcam or camera device. Outputs `VideoFrame` data with configurable *source* device selection."

    """Captures live video from a webcam or camera device."""

    def __init__(self, config: CameraConfig = CameraConfig()) -> None:
        super().__init__(config)

    @classmethod
    def get_options(cls, values: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, str]] = []
        try:
            from cv2_enumerate_cameras import enumerate_cameras

            for cam in enumerate_cameras():
                idx = str(cam.index)
                label = cam.name if cam.name else f"Camera {idx}"
                results.append({"value": idx, "label": label})
        except Exception:
            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    results.append({"value": str(i), "label": f"Camera {i}"})
                    cap.release()

        return {"config": {"source": results}}
