"""ObjectSegmenter conduit — real-time instance segmentation via YOLOE-26."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import (
    ObjectSegmentationFrame,
    TextFrame,
    VideoDataFormat,
    VideoFrame,
)
from src.core.utils import auto_device, resize_and_crop

_YOLO_DIR = Path(__file__).resolve().parent

_MODEL_MAP = {
    "small": str(_YOLO_DIR / "yoloe-26s-seg.pt"),
    "medium": str(_YOLO_DIR / "yoloe-26m-seg.pt"),
    "large": str(_YOLO_DIR / "yoloe-26l-seg.pt"),
    "giant": str(_YOLO_DIR / "yoloe-26x-seg.pt"),
}

_TRACKER_DEFAULTS = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.8,
    "with_reid": False,
    "model": "auto",
}


class ObjectSegmenterConfig(BaseModel):
    model: Literal["small", "medium", "large", "giant"] = "large"
    conf: float = 0.07
    new_track_thresh: float = 0.15
    track_high_thresh: float = 0.15
    track_low_thresh: float = 0.07
    match_thresh: float = 0.8
    device: Literal["auto", "cpu", "cuda", "rocm", "mps"] = "auto"


class ObjectSegmenterInputs(NamedTuple):
    video: Receiver[VideoFrame]
    prompts: Receiver[TextFrame]  # e.g. "head, hand, a pink door"


class ObjectSegmenterOutputs(NamedTuple):
    segmentations: Sender[ObjectSegmentationFrame]
    video: Sender[VideoFrame]


class ObjectSegmenter(ThreadedComponent[ObjectSegmenterInputs, ObjectSegmenterOutputs]):
    description = "Segments objects in video frames"

    tags = Tag(io={"conduit"}, functionality={"vision"}, gpu={"cpu", "nvidia", "apple"})

    def __init__(self, config: ObjectSegmenterConfig) -> None:
        super().__init__()
        self.config = config
        self._model: Any = None
        self._tracker_yaml: str | None = None
        self._prompts: list[str] = []
        self._device = auto_device(config.device)

    def _build_tracker_yaml(self) -> str:
        cfg = dict(_TRACKER_DEFAULTS)
        cfg["track_high_thresh"] = self.config.track_high_thresh
        cfg["track_low_thresh"] = self.config.track_low_thresh
        cfg["new_track_thresh"] = self.config.new_track_thresh
        cfg["track_buffer"] = _TRACKER_DEFAULTS["track_buffer"]
        cfg["match_thresh"] = self.config.match_thresh

        lines = [f"{k}: {v}" for k, v in cfg.items()]
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="tracker_",
            delete=False,
        )
        tmp.write("\n".join(lines) + "\n")
        tmp.close()
        return tmp.name

    def setup(self, outputs: ObjectSegmenterOutputs) -> None:
        from ultralytics import YOLOE

        checkpoint = _MODEL_MAP[self.config.model]
        print(f"[ObjectSegmenter] Loading YOLOE {checkpoint} on {self._device}")

        self._model = YOLOE(checkpoint)

        self._tracker_yaml = self._build_tracker_yaml()

        print("[ObjectSegmenter] Model loaded")

    def _set_phrases(self, phrases: list[str]) -> None:
        phrases = [p.strip() for p in phrases if p and p.strip()]
        dedup: list[str] = []
        seen: set[str] = set()
        for p in phrases:
            if p not in seen:
                dedup.append(p)
                seen.add(p)

        if dedup == self._prompts:
            return
        if not dedup:
            return

        self._prompts = dedup
        prev_cwd = os.getcwd()
        os.chdir(_YOLO_DIR)
        try:
            self._model.set_classes(dedup)
        finally:
            os.chdir(prev_cwd)

    def _infer(
        self, frame: np.ndarray
    ) -> None | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        kwargs: dict[str, Any] = dict(
            imgsz=640,
            conf=self.config.conf,
            iou=0.7,
            rect=False,
            half=False,
            max_det=16,
            retina_masks=False,
            agnostic_nms=False,
            verbose=False,
        )
        if self._device != "auto":
            kwargs["device"] = self._device

        if self._tracker_yaml is not None:
            results = self._model.track(
                frame, persist=True, tracker=self._tracker_yaml, **kwargs
            )
        else:
            results = self._model.predict(frame, **kwargs)

        result = results[0]
        boxes = result.boxes
        masks_result = result.masks

        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        track_ids = None
        if getattr(boxes, "is_track", False) and boxes.id is not None:
            track_ids = boxes.id.int().cpu().numpy()

        mask_data = (
            masks_result.data.cpu().numpy() if masks_result is not None else None
        )

        keep_masks: list[np.ndarray] = []
        keep_boxes: list[np.ndarray] = []
        keep_scores: list[float] = []
        keep_ids: list[int] = []
        keep_labels: list[str] = []

        for i in range(len(cls)):
            c = int(cls[i])
            if c < 0 or c >= len(self._prompts):
                continue
            keep_boxes.append(xyxy[i])
            keep_scores.append(float(conf[i]))
            keep_ids.append(int(track_ids[i]) if track_ids is not None else i)
            keep_labels.append(self._prompts[c])
            if mask_data is not None:
                keep_masks.append(mask_data[i].astype(bool))

        if not keep_boxes:
            return None

        n = len(keep_boxes)
        if keep_masks:
            masks_arr = np.stack(keep_masks)
        else:
            masks_arr = np.zeros((n, 0, 0), dtype=bool)

        return (
            masks_arr,
            np.stack(keep_boxes).astype(np.float32),
            np.array(keep_scores, dtype=np.float32),
            np.array(keep_ids, dtype=np.int64),
            tuple(keep_labels),
        )

    def run(
        self, inputs: ObjectSegmenterInputs, outputs: ObjectSegmenterOutputs
    ) -> None:
        print("[ObjectSegmenter] Starting...")

        inputs.video.newest = True
        inputs.prompts.newest = True
        inputs.prompts.blocking = False

        print("[ObjectSegmenter] Running inference loop")
        for frame in inputs.video:
            if frame is None:
                break

            prompt_frame = next(inputs.prompts, None)
            if prompt_frame is not None:
                new_prompts = [
                    p.strip() for p in prompt_frame.text.split(",") if p.strip()
                ]
                self._set_phrases(new_prompts)

            rgb = frame.get(VideoDataFormat.RGB)
            rgb = resize_and_crop(rgb, 640, 640)

            if not self._prompts:
                continue

            result = self._infer(rgb)

            if result is not None:
                masks, boxes, scores, object_ids, labels = result
                outputs.segmentations.send(
                    ObjectSegmentationFrame.new(
                        masks=masks,
                        boxes=boxes,
                        scores=scores,
                        object_ids=object_ids,
                        labels=labels,
                    )
                )
            else:
                outputs.segmentations.send(
                    ObjectSegmentationFrame.new(
                        masks=np.zeros((0, 0, 0), dtype=bool),
                        boxes=np.zeros((0, 4), dtype=np.float32),
                        scores=np.zeros((0,), dtype=np.float32),
                        object_ids=np.zeros((0,), dtype=np.int64),
                        labels=(),
                    )
                )
            outputs.video.send(VideoFrame.new(data=rgb, format=VideoDataFormat.RGB))

        print("[ObjectSegmenter] Stopped")
