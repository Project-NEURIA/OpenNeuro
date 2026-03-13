"""MonocularDepthEstimator conduit — monocular depth estimation via DA3."""

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


class MonocularDepthEstimatorConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"configOptions": {"model": {}}})

    process_res: int = 504
    output_width: int = 504
    output_height: int = 504
    device: str = "auto"
    model: str = "depth-anything/DA3METRIC-LARGE"


class MonocularDepthEstimatorInputs(NamedTuple):
    video: Receiver[VideoFrame]
    camera_params: Receiver[CameraParamsFrame]


class MonocularDepthEstimatorOutputs(NamedTuple):
    depth: Sender[DepthFrame]
    video: Sender[VideoFrame]


class MonocularDepthEstimator(
    Component[MonocularDepthEstimatorInputs, MonocularDepthEstimatorOutputs]
):
    """Monocular depth estimation backed by Depth Anything 3.

    Consumes VideoFrames and CameraParamsFrames, runs DA3 inference,
    and emits DepthFrames plus the resized/cropped VideoFrame.
    """

    def __init__(self, config: MonocularDepthEstimatorConfig) -> None:
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
            {
                "value": "depth-anything/DA3METRIC-LARGE",
                "label": "DA3 Metric Large (Recommended)",
            },
            {
                "value": "depth-anything/DA3NESTED-GIANT-LARGE",
                "label": "DA3 Nested Giant-Large",
            },
        ]

    def setup(self) -> None:
        import os

        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

        from depth_anything_3.api import DepthAnything3

        print(
            f"[MonocularDepthEstimator] Loading {self.config.model} on {self._device} ({self._dtype})"
        )
        self._model = DepthAnything3.from_pretrained(self.config.model).to(
            device=self._device
        )
        self._model.eval()
        print("[MonocularDepthEstimator] Model loaded")

    def _infer(self, rgb: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Run DA3 inference and return metric depth in metres."""
        import torch
        from depth_anything_3.utils.alignment import apply_metric_scaling

        with torch.inference_mode():
            pred = self._model.inference(
                image=[rgb],
                intrinsics=K[np.newaxis],
                process_res=self.config.process_res,
            )

        raw_depth = pred.depth[0].astype(np.float32)
        if pred.is_metric:
            return raw_depth

        depth_t = torch.from_numpy(raw_depth).float()[None, None]
        K_t = torch.from_numpy(K).float()[None, None]
        return apply_metric_scaling(depth_t, K_t)[0, 0].numpy()

    def _resize_outputs(
        self, depth: np.ndarray, vframe: VideoFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize/crop depth and video to the configured output dimensions."""
        ow, oh = self.config.output_width, self.config.output_height
        return resize_and_crop(depth, ow, oh), resize_and_crop(
            vframe.get(VideoDataFormat.BGR), ow, oh
        )

    def run(
        self,
        inputs: MonocularDepthEstimatorInputs,
        outputs: MonocularDepthEstimatorOutputs,
    ) -> None:
        cam_iter = inputs.camera_params(self, newest=True)

        print("[MonocularDepthEstimator] Running inference loop")
        for vframe in inputs.video(self, newest=True):
            if vframe is None:
                break

            cam_frame = next(cam_iter, None)
            if cam_frame is None:
                break

            rgb = vframe.get(VideoDataFormat.RGB)
            K = cam_frame.intrinsics.astype(np.float32)

            depth_m = self._infer(rgb, K)
            depth_out, video_out = self._resize_outputs(depth_m, vframe)

            outputs.depth.send(DepthFrame.new(data=depth_out, is_metric=True))
            outputs.video.send(
                VideoFrame.new(data=video_out, format=VideoDataFormat.BGR)
            )

        print("[MonocularDepthEstimator] Stopped")
