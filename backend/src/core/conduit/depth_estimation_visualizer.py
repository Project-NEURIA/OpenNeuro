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

            # Normalize depth to 0-255 using robust percentiles
            d = depth_frame.data
            valid = np.isfinite(d) & (d > 0)
            if np.any(valid):
                lo = float(np.percentile(d[valid], 2))
                hi = float(np.percentile(d[valid], 98))
            else:
                lo, hi = 0.0, 1.0
            norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            depth_u8 = (norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

            img = video.get(VideoDataFormat.BGR).copy()

            # Resize depth colormap to match video if dimensions differ
            if depth_color.shape[:2] != img.shape[:2]:
                depth_color = cv2.resize(
                    depth_color,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            overlay = cv2.addWeighted(img, 1.0 - _ALPHA, depth_color, _ALPHA, 0.0)
            outputs.video.send(VideoFrame.new(data=overlay, format=VideoDataFormat.BGR))

        print("[DepthEstimationVisualizer] Stopped")
