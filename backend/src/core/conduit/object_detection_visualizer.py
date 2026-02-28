"""ObjectDetectionVisualizer conduit — composites bounding boxes onto video frames."""

from __future__ import annotations

import threading
from typing import NamedTuple

import cv2
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import ObjectDetectionFrame, VideoDataFormat, VideoFrame

# Fixed BGR color palette, cycled per prompt index
_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]


class ObjectDetectionVisualizerConfig(BaseModel):
    score_threshold: float = 0.5
    line_thickness: int = 2
    font_scale: float = 0.5


class ObjectDetectionVisualizerInputs(NamedTuple):
    detections: Receiver[ObjectDetectionFrame]
    video: Receiver[VideoFrame]


class ObjectDetectionVisualizerOutputs(NamedTuple):
    video: Sender[VideoFrame]


class ObjectDetectionVisualizer(
    Component[ObjectDetectionVisualizerInputs, ObjectDetectionVisualizerOutputs]
):
    """Composites bounding boxes and labels onto video frames."""

    def __init__(self, config: ObjectDetectionVisualizerConfig) -> None:
        super().__init__()
        self.config = config
        self._latest_frame: VideoFrame | None = None
        self._lock = threading.Lock()

    def _video_consumer(self, inputs: ObjectDetectionVisualizerInputs) -> None:
        """Daemon thread: keeps latest video frame up to date."""
        for frame in inputs.video(self, newest=True):
            if frame is None:
                break
            with self._lock:
                self._latest_frame = frame

    def run(
        self,
        inputs: ObjectDetectionVisualizerInputs,
        outputs: ObjectDetectionVisualizerOutputs,
    ) -> None:
        print("[ObjectDetectionVisualizer] Starting")

        video_thread = threading.Thread(
            target=self._video_consumer, args=(inputs,), daemon=True
        )
        video_thread.start()

        for det_frame in inputs.detections(self):
            if det_frame is None:
                break

            with self._lock:
                video = self._latest_frame

            if video is None:
                continue

            img = video.get(VideoDataFormat.BGR).copy()

            for pi, prompt in enumerate(det_frame.prompts):
                color = _COLORS[pi % len(_COLORS)]
                for si in range(det_frame.scores.shape[1]):
                    score = float(det_frame.scores[pi, si])
                    if score < self.config.score_threshold:
                        continue
                    x1, y1, x2, y2 = det_frame.boxes[pi, si].astype(int)
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.line_thickness,
                    )
                    label = f"{prompt}: {score:.0%}"
                    (tw, th), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale,
                        1,
                    )
                    cv2.rectangle(
                        img,
                        (x1, y1 - th - 4),
                        (x1 + tw, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            outputs.video.send(VideoFrame.new(data=img, format=VideoDataFormat.BGR))

        video_thread.join(timeout=2.0)
        print("[ObjectDetectionVisualizer] Stopped")
