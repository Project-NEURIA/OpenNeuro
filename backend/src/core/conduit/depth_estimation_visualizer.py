"""DepthEstimationVisualizer conduit — colorized depth overlay on video frames."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import DepthFrame, VideoDataFormat, VideoFrame

_ALPHA = 0.5


class DepthEstimationVisualizerInputs(NamedTuple):
    depth: Receiver[DepthFrame]
    video: Receiver[VideoFrame]


class DepthEstimationVisualizerOutputs(NamedTuple):
    video: Sender[VideoFrame]


class DepthEstimationVisualizer(
    Component[DepthEstimationVisualizerInputs, DepthEstimationVisualizerOutputs]
):
    """Colorizes depth and overlays it on video frames."""

    @staticmethod
    def _colorize_depth(depth_data: np.ndarray) -> np.ndarray:
        """Colorize metric depth via inverse-depth so close=red, far=blue."""
        valid = np.isfinite(depth_data) & (depth_data > 0)
        inv = np.zeros_like(depth_data)
        inv[valid] = 1.0 / depth_data[valid]
        if np.any(valid):
            lo = float(inv[valid].min())
            hi = float(inv[valid].max())
        else:
            lo, hi = 0.0, 1.0
        norm = np.zeros_like(depth_data)
        norm[valid] = np.clip((inv[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        depth_u8 = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return colored

    @staticmethod
    def _overlay(img: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
        """Resize depth colormap to match the image and blend them together."""
        if depth_color.shape[:2] != img.shape[:2]:
            depth_color = cv2.resize(
                depth_color,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.addWeighted(img, 1.0 - _ALPHA, depth_color, _ALPHA, 0.0)

    def run(
        self,
        inputs: DepthEstimationVisualizerInputs,
        outputs: DepthEstimationVisualizerOutputs,
    ) -> None:
        print("[DepthEstimationVisualizer] Starting")

        video_iter = inputs.video(self)
        for depth_frame in inputs.depth(self):
            if depth_frame is None:
                break

            video = next(video_iter, None)
            if video is None:
                break

            depth_color = self._colorize_depth(depth_frame.data)
            img = video.get(VideoDataFormat.BGR).copy()
            overlay = self._overlay(img, depth_color)
            outputs.video.send(VideoFrame.new(data=overlay, format=VideoDataFormat.BGR))

        print("[DepthEstimationVisualizer] Stopped")
