"""ObjectDetectionVisualizer conduit — composites bounding boxes onto video frames."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent, Tag
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

# Can be made as configs, but not really necessary
_SCORE_THRESHOLD = 0.5
_LINE_THICKNESS = 2
_FONT_SCALE = 0.5


def _draw_labeled_box(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, _LINE_THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(
        img,
        label,
        (x1, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        _FONT_SCALE,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_detections(img: np.ndarray, det_frame: ObjectDetectionFrame) -> None:
    for pi, prompt in enumerate(det_frame.prompts):
        color = _COLORS[pi % len(_COLORS)]
        for si in range(det_frame.scores.shape[1]):
            score = float(det_frame.scores[pi, si])
            if score < _SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = det_frame.boxes[pi, si].astype(int)
            _draw_labeled_box(img, x1, y1, x2, y2, f"{prompt}: {score:.0%}", color)


class ObjectDetectionVisualizerInputs(NamedTuple):
    detections: Receiver[ObjectDetectionFrame]
    video: Receiver[VideoFrame]


class ObjectDetectionVisualizerOutputs(NamedTuple):
    video: Sender[VideoFrame]


class ObjectDetectionVisualizer(
    PrimitiveComponent[
        ObjectDetectionVisualizerInputs, ObjectDetectionVisualizerOutputs
    ]
):
    """Composites bounding boxes and labels onto video frames."""

    _tags = Tag(io={"conduit"}, functionality={"image"})

    def run(
        self,
        inputs: ObjectDetectionVisualizerInputs,
        outputs: ObjectDetectionVisualizerOutputs,
    ) -> None:
        print("[ObjectDetectionVisualizer] Starting")

        video_iter = inputs.video(self)
        for det_frame in inputs.detections(self):
            if det_frame is None:
                break

            video = next(video_iter, None)
            if video is None:
                break

            img = video.get(VideoDataFormat.BGR).copy()
            _draw_detections(img, det_frame)
            outputs.video.send(VideoFrame.new(data=img, format=VideoDataFormat.BGR))

        print("[ObjectDetectionVisualizer] Stopped")
