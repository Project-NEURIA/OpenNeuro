"""ObjectDetector conduit — real-time object detection via SAM3."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, NamedTuple, Any

import numpy as np
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import (
    ObjectDetectionFrame,
    TextFrame,
    VideoDataFormat,
    VideoFrame,
)
from src.core.utils import auto_device, auto_dtype, center_crop_and_resize, to_numpy

MODEL_ID = "facebook/sam3"


class ObjectDetectorConfig(BaseModel):
    resolution: int = 336
    det_thresh: float = 0.5
    new_det_thresh: float = 0.85
    max_tracking_per_object: int = 4
    session_ttl: int = 400
    device: str = "auto"


class ObjectDetectorInputs(NamedTuple):
    video: Receiver[VideoFrame]
    prompts: Receiver[TextFrame]  # e.g. "head, hand, a pink door"


class ObjectDetectorOutputs(NamedTuple):
    detections: Sender[ObjectDetectionFrame]
    video: Sender[VideoFrame]


class ObjectDetector(Component[ObjectDetectorInputs, ObjectDetectorOutputs]):
    """Text-prompted object detector backed by SAM3.

    Consumes VideoFrames, runs SAM3 inference, and emits
    ObjectDetectionFrames plus the processed VideoFrame.
    """

    def __init__(self, config: ObjectDetectorConfig) -> None:
        super().__init__()
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._session: Any = None
        self._session_age = 0
        self._prompts: list[str] = []
        self._lock = threading.Lock()
        self._device = auto_device(config.device)
        self._dtype = auto_dtype(self._device)

    def _ensure_model(self) -> None:
        """Lazy-load SAM3 model and processor."""
        if self._model is not None:
            return

        from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

        resolution = self.config.resolution

        print(f"[ObjectDetector] Loading SAM3 on {self._device} ({self._dtype})")

        config = Sam3VideoConfig.from_pretrained(MODEL_ID)
        config.image_size = resolution
        config.score_threshold_detection = self.config.det_thresh
        config.new_det_thresh = self.config.new_det_thresh
        config.max_num_objects = (
            len(self._prompts) * self.config.max_tracking_per_object
        )

        self._model = Sam3VideoModel.from_pretrained(
            MODEL_ID,
            config=config,
        ).to(self._device, self._dtype)
        self._model.eval()

        self._processor = Sam3VideoProcessor.from_pretrained(
            MODEL_ID,
            target_size=resolution,
        )
        self._processor.image_processor.size = {
            "height": resolution,
            "width": resolution,
        }

        self._init_session()
        print("[ObjectDetector] Model loaded")

    def _init_session(self) -> None:
        """Create a new SAM3 tracking session with current prompts."""
        self._session = self._processor.init_video_session(
            inference_device=self._device,
            inference_state_device=self._device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self._dtype,
        )
        if self._prompts:
            self._processor.add_text_prompt(
                inference_session=self._session,
                text=self._prompts,
            )
        self._session_age = 0

    def _gather_by_prompt(self, processed: dict) -> tuple[np.ndarray, np.ndarray]:
        """Reshape flat detections into (N, M) arrays keyed by prompt.

        Returns (boxes, scores) with shapes (N, M, 4) and (N, M).
        """
        import torch

        obj_ids = to_numpy(processed["object_ids"])
        all_scores = to_numpy(processed["scores"])
        all_boxes = to_numpy(processed["boxes"])

        prompt_to_idx = {p: i for i, p in enumerate(self._prompts)}
        oid_to_prompt_idx: dict[int, int] = {}
        for prompt, ids in (processed.get("prompt_to_obj_ids") or {}).items():
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().tolist()
            idx = prompt_to_idx.get(str(prompt))
            if idx is not None:
                for oid in ids:
                    oid_to_prompt_idx[int(oid)] = idx

        N = len(self._prompts)
        M = self.config.max_tracking_per_object
        boxes = np.zeros((N, M, 4), dtype=np.float32)
        scores = np.zeros((N, M), dtype=np.float32)
        slot = [0] * N

        order = np.argsort(-all_scores)
        for i in order:
            prompt_idx = oid_to_prompt_idx.get(int(obj_ids[i]))
            if prompt_idx is None or slot[prompt_idx] >= M:
                continue
            slot_idx = slot[prompt_idx]
            boxes[prompt_idx, slot_idx] = all_boxes[i]
            scores[prompt_idx, slot_idx] = all_scores[i]
            slot[prompt_idx] += 1

        return boxes, scores

    def _purge_dead_tracklets(self, outputs: dict) -> None:
        """Remove objects whose masks are NaN or all-negative."""
        masks = outputs["obj_id_to_mask"]
        scores = outputs["obj_id_to_score"]
        for oid in [
            oid for oid, m in masks.items() if m.isnan().any() or (m <= 0).all()
        ]:
            self._session.remove_object(oid, strict=False)
            masks.pop(oid, None)
            scores.pop(oid, None)

    def _prune_old_frames(self) -> None:
        """Null out consumed frame tensors to bound memory."""
        pf = self._session.processed_frames
        if pf is not None and len(pf) > 1:
            latest = max(pf.keys())
            for k in list(pf.keys()):
                if k < latest:
                    pf[k] = None

    def _prompt_listener(self, inputs: ObjectDetectorInputs) -> None:
        """Daemon thread: consume prompt updates and reinitialize session."""
        for frame in inputs.prompts(self):
            if frame is None:
                break
            new_prompts = [p.strip() for p in frame.text.split(",") if p.strip()]
            if new_prompts == self._prompts:
                continue
            print(f"[ObjectDetector] Prompts updated: {new_prompts}")
            with self._lock:
                self._prompts = new_prompts
                self._init_session()

    def run(self, inputs: ObjectDetectorInputs, outputs: ObjectDetectorOutputs) -> None:
        import torch

        print("[ObjectDetector] Starting, waiting for prompts before loading model...")

        prompt_thread = threading.Thread(
            target=self._prompt_listener, args=(inputs,), daemon=True
        )
        prompt_thread.start()

        print("[ObjectDetector] Running inference loop")
        for frame in inputs.video(self, newest=True):
            if frame is None:
                break

            rgb = frame.get(VideoDataFormat.RGB)
            res = self.config.resolution
            cropped = center_crop_and_resize(rgb, res, res)

            if not self._prompts:
                continue

            self._ensure_model()

            with self._lock, torch.inference_mode():
                inputs_processed = self._processor(images=cropped, return_tensors="pt")
                pixel_frame = inputs_processed.pixel_values[0].to(
                    self._device, self._dtype
                )
                inputs_processed = self._processor(
                    images=cropped,
                    device=self._device,
                    return_tensors="pt",
                )

                model_outputs = self._model(
                    inference_session=self._session,
                    frame=pixel_frame,
                    reverse=False,
                )

                self._purge_dead_tracklets(model_outputs)
                self._prune_old_frames()

                processed = self._processor.postprocess_outputs(
                    self._session,
                    model_outputs,
                    original_sizes=inputs_processed.original_sizes,
                )

                boxes, scores = self._gather_by_prompt(processed)
                prompts = tuple(self._prompts)

                self._session_age += 1
                if self._session_age >= self.config.session_ttl:
                    self._init_session()

            outputs.detections.send(
                ObjectDetectionFrame.new(
                    boxes=boxes,
                    scores=scores,
                    prompts=prompts,
                )
            )
            outputs.video.send(VideoFrame.new(data=cropped, format=VideoDataFormat.RGB))

        prompt_thread.join(timeout=2.0)
        print("[ObjectDetector] Stopped")
