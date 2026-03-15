"""StereoDepthEstimator conduit — stereo depth estimation via Fast-FoundationStereo."""

from __future__ import annotations

import pathlib
from typing import Any, NamedTuple

import numpy as np
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent, Tag
from src.core.frames import (
    DepthFrame,
    StereoCameraParamsFrame,
    StereoVideoFrame,
    VideoDataFormat,
)
from src.core.utils import auto_device, resize_and_crop

_WEIGHTS_DIR = pathlib.Path(__file__).parent / "weights"
_DEFAULT_MODEL = _WEIGHTS_DIR / "23-36-37" / "model_best_bp2_serialize.pth"


class StereoDepthEstimatorConfig(BaseModel):
    resize_width: int = 512
    resize_height: int = 512
    valid_iters: int = 4
    max_disp: int = 192
    device: str = "auto"


class StereoDepthEstimatorInputs(NamedTuple):
    stereo_video: Receiver[StereoVideoFrame]
    camera_params: Receiver[StereoCameraParamsFrame]


class StereoDepthEstimatorOutputs(NamedTuple):
    depth: Sender[DepthFrame]
    stereo_video: Sender[StereoVideoFrame]


class StereoDepthEstimator(
    PrimitiveComponent[StereoDepthEstimatorInputs, StereoDepthEstimatorOutputs]
):
    """Stereo depth estimation backed by Fast-FoundationStereo.

    Consumes StereoVideoFrames and StereoCameraParamsFrames, runs FFS
    disparity inference, converts to metric depth, and emits DepthFrames
    plus the resized/cropped StereoVideoFrame.
    """

    _tags = Tag(io={"conduit"}, functionality={"image"}, gpu={"cpu", "nvidia", "apple"})

    def __init__(self, config: StereoDepthEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        self._model: Any = None
        self._device = auto_device(config.device)

        # Pre-allocated GPU buffers (created on first frame).
        self._t0_buf: Any = None
        self._t1_buf: Any = None
        self._padder: Any = None
        self._buf_h: int = 0
        self._buf_w: int = 0

    def setup(self) -> None:
        import torch

        print(f"[StereoDepthEstimator] Loading FFS model on {self._device}")
        self._model = torch.load(
            str(_DEFAULT_MODEL), map_location="cpu", weights_only=False
        )
        self._model.args.valid_iters = self.config.valid_iters
        self._model.args.max_disp = self.config.max_disp
        self._model.to(self._device).eval()
        torch.set_grad_enabled(False)
        print("[StereoDepthEstimator] Model loaded")

    def _ensure_buffers(self, H: int, W: int) -> None:
        """Allocate or re-allocate GPU tensors when resolution changes."""
        if self._t0_buf is not None and H == self._buf_h and W == self._buf_w:
            return

        import torch
        from core.utils.utils import InputPadder

        self._buf_h, self._buf_w = H, W
        self._t0_buf = torch.empty(1, 3, H, W, dtype=torch.float32, device=self._device)
        self._t1_buf = torch.empty(1, 3, H, W, dtype=torch.float32, device=self._device)
        self._padder = InputPadder(self._t0_buf.shape, divis_by=32, force_square=False)

    @staticmethod
    def _resize_for_inference(img: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize to cover target dims (INTER_AREA for clean downscaling), then center-crop."""
        import cv2

        h_src, w_src = img.shape[:2]
        scale = max(width / w_src, height / h_src)
        w_s = int(round(w_src * scale))
        h_s = int(round(h_src * scale))
        resized = cv2.resize(img, (w_s, h_s), interpolation=cv2.INTER_AREA)
        y0 = (h_s - height) // 2
        x0 = (w_s - width) // 2
        return resized[y0 : y0 + height, x0 : x0 + width]

    def _infer(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Run FFS inference on resize_width x resize_height images and return disparity."""
        import torch

        rw, rh = self.config.resize_width, self.config.resize_height
        left_rgb = self._resize_for_inference(left_rgb, rw, rh)
        right_rgb = self._resize_for_inference(right_rgb, rw, rh)

        H, W = left_rgb.shape[:2]
        self._ensure_buffers(H, W)

        self._t0_buf.copy_(
            torch.as_tensor(left_rgb, device=self._device)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        self._t1_buf.copy_(
            torch.as_tensor(right_rgb, device=self._device)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        t0_pad, t1_pad = self._padder.pad(self._t0_buf, self._t1_buf)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            disp = self._model.forward(
                t0_pad,
                t1_pad,
                iters=self.config.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

        disp = self._padder.unpad(disp.float())
        return disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    def _resize_outputs(
        self,
        stereo_frame: StereoVideoFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize/crop both video eyes to the configured dimensions."""
        rw, rh = self.config.resize_width, self.config.resize_height
        import cv2

        left_bgr = cv2.cvtColor(stereo_frame.left, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(stereo_frame.right, cv2.COLOR_RGB2BGR)
        return (
            resize_and_crop(left_bgr, rw, rh),
            resize_and_crop(right_bgr, rw, rh),
        )

    def run(
        self,
        inputs: StereoDepthEstimatorInputs,
        outputs: StereoDepthEstimatorOutputs,
    ) -> None:
        cam_iter = inputs.camera_params(self, newest=True)

        print("[StereoDepthEstimator] Running inference loop")
        for stereo_frame in inputs.stereo_video(self, newest=True):
            if stereo_frame is None:
                break

            cam_frame = next(cam_iter, None)
            if cam_frame is None:
                break

            disp = self._infer(stereo_frame.left, stereo_frame.right)

            # Disparity → metric depth.
            K = cam_frame.intrinsics.astype(np.float32)
            _, src_w = stereo_frame.left.shape[:2]
            fx_scaled = K[0, 0] * self.config.resize_width / src_w
            depth_m = (fx_scaled * cam_frame.baseline / np.maximum(disp, 1e-6)).astype(
                np.float32
            )

            left_out, right_out = self._resize_outputs(stereo_frame)

            outputs.depth.send(DepthFrame.new(data=depth_m, is_metric=True))
            outputs.stereo_video.send(
                StereoVideoFrame.new(
                    left=left_out,
                    right=right_out,
                    format=VideoDataFormat.BGR,
                )
            )

        print("[StereoDepthEstimator] Stopped")
