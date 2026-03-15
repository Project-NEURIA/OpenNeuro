"""ObjectLocatorVisualizer conduit — annotates video with 3D object locations."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent
from src.core.frames import ObjectLocationFrame, VideoDataFormat, VideoFrame

# Fixed BGR colour palette, cycled per prompt.
_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]

_SCORE_THRESHOLD = 0.4
_LINE_THICKNESS = 2
_FONT_SCALE = 0.5


def _draw_locations(img: np.ndarray, loc: ObjectLocationFrame) -> None:
    """Draw bounding boxes with 3D coordinate labels onto *img* in-place."""
    h, w = img.shape[:2]

    prompt_color: dict[str, tuple[int, int, int]] = {}
    for label in loc.labels:
        if label not in prompt_color:
            prompt_color[label] = _COLORS[len(prompt_color) % len(_COLORS)]

    for i in range(len(loc.labels)):
        score = float(loc.scores[i])
        if score < _SCORE_THRESHOLD:
            continue

        pos = loc.positions[i]
        if not np.all(np.isfinite(pos)):
            continue

        color = prompt_color[loc.labels[i]]

        # Bounding box (clamped to image bounds).
        x1, y1, x2, y2 = loc.boxes[i].astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, _LINE_THICKNESS)

        # Label showing world-frame coordinates.
        x_w, y_w, z_w = float(pos[0]), float(pos[1]), float(pos[2])
        text = f"{loc.labels[i]} ({x_w:.2f}, {y_w:.2f}, {z_w:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            img,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            _FONT_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


class ObjectLocatorVisualizerInputs(NamedTuple):
    locations: Receiver[ObjectLocationFrame]
    video: Receiver[VideoFrame]


class ObjectLocatorVisualizerOutputs(NamedTuple):
    video: Sender[VideoFrame]


class ObjectLocatorVisualizer(
    PrimitiveComponent[ObjectLocatorVisualizerInputs, ObjectLocatorVisualizerOutputs]
):
    """Draws bounding boxes with 3D world coordinate labels onto video frames."""

    def run(
        self,
        inputs: ObjectLocatorVisualizerInputs,
        outputs: ObjectLocatorVisualizerOutputs,
    ) -> None:
        video_iter = inputs.video(self)
        for loc_frame in inputs.locations(self):
            if loc_frame is None:
                break

            video = next(video_iter, None)
            if video is None:
                break

            img = video.get(VideoDataFormat.BGR).copy()
            _draw_locations(img, loc_frame)
            outputs.video.send(VideoFrame.new(data=img, format=VideoDataFormat.BGR))
