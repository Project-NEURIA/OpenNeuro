"""DepthEstimator conduit — monocular depth estimation via DA3."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import (
    CameraParamsFrame,
    DepthFrame,
    VideoDataFormat,
    VideoFrame,
)
from src.core.utils import auto_device, auto_dtype, resize_and_crop


class DepthEstimatorConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"configOptions": {"model": {}}})

    output_width: int = 640
    output_height: int = 480
    process_res: int = 504
    device: str = "auto"
    model: str = "depth-anything/DA3METRIC-LARGE"


class DepthEstimatorInputs(NamedTuple):
    video: Receiver[VideoFrame]
    camera_params: Receiver[CameraParamsFrame]


class DepthEstimatorOutputs(NamedTuple):
    depth: Sender[DepthFrame]
    video: Sender[VideoFrame]


class DepthEstimator(Component[DepthEstimatorInputs, DepthEstimatorOutputs]):
    """Monocular depth estimation backed by Depth Anything 3.

    Consumes VideoFrames and CameraParamsFrames, runs DA3 inference,
    and emits DepthFrames plus the resized/cropped VideoFrame.
    """

    def __init__(self, config: DepthEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        self._model: Any = None
        self._device = auto_device(config.device)
        self._dtype = auto_dtype(self._device)

    @classmethod
    def get_config_options(
        cls, field: str, values: dict[str, Any] | None = None
    ) -> list[dict[str, str]] | None:
        if field != "config.model":
            return None
        return [
            {"value": "depth-anything/DA3METRIC-LARGE", "label": "DA3 Metric Large"},
            {
                "value": "depth-anything/DA3NESTED-GIANT-LARGE",
                "label": "DA3 Nested Giant-Large",
            },
        ]

    def _ensure_model(self) -> None:
        """Lazy-load DA3 model on first use."""
        if self._model is not None:
            return

        import os

        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

        from depth_anything_3.api import DepthAnything3

        print(
            f"[DepthEstimator] Loading {self.config.model} on {self._device} ({self._dtype})"
        )
        self._model = DepthAnything3.from_pretrained(self.config.model).to(
            device=self._device
        )
        self._model.eval()
        print("[DepthEstimator] Model loaded")

    def run(self, inputs: DepthEstimatorInputs, outputs: DepthEstimatorOutputs) -> None:
        import torch
        from depth_anything_3.utils.alignment import apply_metric_scaling

        self._ensure_model()
        cam_iter = inputs.camera_params(self, newest=True)

        print("[DepthEstimator] Running inference loop")
        for vframe in inputs.video(self, newest=True):
            if vframe is None:
                break

            cam_frame = next(cam_iter, None)
            if cam_frame is None:
                break

            rgb = vframe.get(VideoDataFormat.RGB)
            K = cam_frame.intrinsics.astype(np.float32)

            with torch.inference_mode():
                pred = self._model.inference(
                    image=[rgb],
                    intrinsics=K[np.newaxis],
                    process_res=self.config.process_res,
                )

            raw_depth = pred.depth[0].astype(np.float32)

            if pred.is_metric:
                depth_m = raw_depth
            else:
                depth_t = torch.from_numpy(raw_depth).float()[None, None]
                K_t = torch.from_numpy(K).float()[None, None]
                depth_m = apply_metric_scaling(depth_t, K_t)[0, 0].numpy()

            ow, oh = self.config.output_width, self.config.output_height
            depth_out = resize_and_crop(depth_m, ow, oh)
            bgr = vframe.get(VideoDataFormat.BGR)
            video_out = resize_and_crop(bgr, ow, oh)

            outputs.depth.send(DepthFrame.new(data=depth_out, is_metric=True))
            outputs.video.send(
                VideoFrame.new(data=video_out, format=VideoDataFormat.BGR)
            )

        print("[DepthEstimator] Stopped")
