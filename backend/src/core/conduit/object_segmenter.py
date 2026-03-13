"""ObjectSegmenter conduit — real-time instance segmentation via SAM3."""

from __future__ import annotations

import threading
from typing import NamedTuple, Any

import numpy as np
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import (
    ObjectSegmentationFrame,
    TextFrame,
    VideoDataFormat,
    VideoFrame,
)
from src.core.utils import auto_device, auto_dtype, resize_and_crop, to_numpy

MODEL_ID = "facebook/sam3"


class ObjectSegmenterConfig(BaseModel):
    resolution: int = 336
    det_thresh: float = 0.5
    new_det_thresh: float = 0.85
    max_per_prompt: int = 6
    score_thresh: float = 0.4
    session_ttl: int = 400
    device: str = "auto"


class ObjectSegmenterInputs(NamedTuple):
    video: Receiver[VideoFrame]
    prompts: Receiver[TextFrame]  # e.g. "head, hand, a pink door"


class ObjectSegmenterOutputs(NamedTuple):
    segmentations: Sender[ObjectSegmentationFrame]
    video: Sender[VideoFrame]


class ObjectSegmenter(Component[ObjectSegmenterInputs, ObjectSegmenterOutputs]):
    """Text-prompted instance segmenter backed by SAM3.

    Consumes VideoFrames, runs SAM3 inference, and emits
    ObjectSegmentationFrames (with masks) plus the processed VideoFrame.
    """

    def __init__(self, config: ObjectSegmenterConfig) -> None:
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

        print(f"[ObjectSegmenter] Loading SAM3 on {self._device} ({self._dtype})")

        config = Sam3VideoConfig.from_pretrained(MODEL_ID)
        config.image_size = resolution
        config.score_threshold_detection = self.config.det_thresh
        config.new_det_thresh = self.config.new_det_thresh

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
        print("[ObjectSegmenter] Model loaded")

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

    def _gather_flat(
        self, processed: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        """Extract flat arrays of masks, boxes, scores, object_ids, labels.

        Filters by score_thresh and limits max_per_prompt (highest scores kept).
        """
        import torch

        obj_ids = to_numpy(processed["object_ids"])
        all_scores = to_numpy(processed["scores"])
        all_boxes = to_numpy(processed["boxes"])
        all_masks = to_numpy(processed["masks"])

        # Map object IDs to prompt labels
        oid_to_prompt: dict[int, str] = {}
        prompt_set = set(self._prompts)
        for prompt, ids in (processed.get("prompt_to_obj_ids") or {}).items():
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().tolist()
            prompt_str = str(prompt)
            if prompt_str in prompt_set:
                for oid in ids:
                    oid_to_prompt[int(oid)] = prompt_str

        # Score-filter and per-prompt limit
        order = np.argsort(-all_scores)
        prompt_counts: dict[str, int] = {}
        keep_indices: list[int] = []
        for i in order:
            score = float(all_scores[i])
            if score < self.config.score_thresh:
                continue
            label = oid_to_prompt.get(int(obj_ids[i]))
            if label is None:
                continue
            count = prompt_counts.get(label, 0)
            if count >= self.config.max_per_prompt:
                continue
            prompt_counts[label] = count + 1
            keep_indices.append(int(i))

        if not keep_indices:
            h, w = all_masks.shape[1:3] if all_masks.ndim >= 3 else (0, 0)
            return (
                np.zeros((0, h, w), dtype=bool),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                (),
            )

        idx = np.array(keep_indices)
        labels = tuple(oid_to_prompt[int(obj_ids[i])] for i in idx)
        return (
            all_masks[idx].astype(bool),
            all_boxes[idx],
            all_scores[idx],
            obj_ids[idx].astype(np.int64),
            labels,
        )

    def _purge_dead_tracklets(self, outputs: dict[str, Any]) -> None:
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

    def _prompt_listener(self, inputs: ObjectSegmenterInputs) -> None:
        """Daemon thread: consume prompt updates and reinitialize session."""
        for frame in inputs.prompts(self):
            if frame is None:
                break
            new_prompts = [p.strip() for p in frame.text.split(",") if p.strip()]
            if new_prompts == self._prompts:
                continue
            print(f"[ObjectSegmenter] Prompts updated: {new_prompts}")
            with self._lock:
                self._prompts = new_prompts
                self._init_session()

    def run(
        self, inputs: ObjectSegmenterInputs, outputs: ObjectSegmenterOutputs
    ) -> None:
        import torch

        print("[ObjectSegmenter] Starting, waiting for prompts before loading model...")

        prompt_thread = threading.Thread(
            target=self._prompt_listener, args=(inputs,), daemon=True
        )
        prompt_thread.start()

        print("[ObjectSegmenter] Running inference loop")
        for frame in inputs.video(self, newest=True):
            if frame is None:
                break

            rgb = frame.get(VideoDataFormat.RGB)
            res = self.config.resolution
            cropped = resize_and_crop(rgb, res, res)

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

                masks, boxes, scores, object_ids, labels = self._gather_flat(processed)
                self._session_age += 1
                if self._session_age >= self.config.session_ttl:
                    self._init_session()

            outputs.segmentations.send(
                ObjectSegmentationFrame.new(
                    masks=masks,
                    boxes=boxes,
                    scores=scores,
                    object_ids=object_ids,
                    labels=labels,
                )
            )
            outputs.video.send(VideoFrame.new(data=cropped, format=VideoDataFormat.RGB))

        prompt_thread.join(timeout=2.0)
        print("[ObjectSegmenter] Stopped")
