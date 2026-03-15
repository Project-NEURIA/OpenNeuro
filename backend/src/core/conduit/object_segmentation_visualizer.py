"""ObjectSegmentationVisualizer conduit — composites instance masks, boxes, and labels onto video."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent, Tag
from src.core.frames import ObjectSegmentationFrame, VideoDataFormat, VideoFrame

# Fixed BGR color palette, cycled per prompt
_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]

_ALPHA = 0.45
_SCORE_THRESHOLD = 0.4
_LINE_THICKNESS = 2
_FONT_SCALE = 0.5


def _draw_segmentations(img: np.ndarray, seg: ObjectSegmentationFrame) -> None:
    """Draw instance masks, bounding boxes, and labels onto *img* in-place."""
    h, w = img.shape[:2]

    # Assign a color index per unique prompt label
    prompt_color: dict[str, tuple[int, int, int]] = {}
    for label in seg.labels:
        if label not in prompt_color:
            prompt_color[label] = _COLORS[len(prompt_color) % len(_COLORS)]

    for i in range(len(seg.labels)):
        score = float(seg.scores[i])
        if score < _SCORE_THRESHOLD:
            continue

        color = prompt_color[seg.labels[i]]

        # Alpha-blend mask onto image
        mask = seg.masks[i]
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        color_arr = np.array(color, dtype=np.float32)
        img[mask] = (
            img[mask].astype(np.float32) * (1.0 - _ALPHA) + color_arr * _ALPHA
        ).astype(np.uint8)

        # Bounding box
        x1, y1, x2, y2 = seg.boxes[i].astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, _LINE_THICKNESS)

        # Label with score and object ID
        text = f"{seg.labels[i]}:{score:.0%} id={int(seg.object_ids[i])}"
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


class ObjectSegmentationVisualizerInputs(NamedTuple):
    segmentations: Receiver[ObjectSegmentationFrame]
    video: Receiver[VideoFrame]


class ObjectSegmentationVisualizerOutputs(NamedTuple):
    video: Sender[VideoFrame]


class ObjectSegmentationVisualizer(
    PrimitiveComponent[
        ObjectSegmentationVisualizerInputs, ObjectSegmentationVisualizerOutputs
    ]
):
    """Composites instance masks, bounding boxes, and labels onto video frames."""

    _tags = Tag(io={"conduit"}, functionality={"image"})

    def run(
        self,
        inputs: ObjectSegmentationVisualizerInputs,
        outputs: ObjectSegmentationVisualizerOutputs,
    ) -> None:
        print("[ObjectSegmentationVisualizer] Starting")

        video_iter = inputs.video(self)
        for seg_frame in inputs.segmentations(self):
            if seg_frame is None:
                break

            video = next(video_iter, None)
            if video is None:
                break

            img = video.get(VideoDataFormat.BGR).copy()
            _draw_segmentations(img, seg_frame)
            outputs.video.send(VideoFrame.new(data=img, format=VideoDataFormat.BGR))

        print("[ObjectSegmentationVisualizer] Stopped")
