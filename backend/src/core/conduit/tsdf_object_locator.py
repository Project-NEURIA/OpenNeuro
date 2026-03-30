"""TSDFObjectLocator conduit — 3D object localisation using TSDF-rendered depth."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
import torch
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Tag, ThreadedComponent
from src.core.frames import (
    CameraParamsFrame,
    DepthFrame,
    ObjectLocationFrame,
    ObjectSegmentationFrame,
)
from src.core.utils import auto_device


class TSDFObjectLocatorConfig(BaseModel):
    depth_tolerance: float = 0.3
    min_pixels: int = 50
    weiszfeld_iters: int = 20
    gate_with_live_depth: bool = True


class TSDFObjectLocatorInputs(NamedTuple):
    segmentations: Receiver[ObjectSegmentationFrame]
    tsdf_depth: Receiver[DepthFrame]
    camera_params: Receiver[CameraParamsFrame]
    live_depth: Receiver[DepthFrame] | None = None


class TSDFObjectLocatorOutputs(NamedTuple):
    locations: Sender[ObjectLocationFrame]


def _geometric_median(points: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Weiszfeld iterative algorithm for the geometric (L1) median."""
    center = points.median(dim=0).values
    for _ in range(num_iters):
        dists = torch.linalg.norm(points - center, dim=1, keepdim=True).clamp(min=1e-8)
        w = 1.0 / dists
        center = (points * w).sum(0) / w.sum()
    return center


class TSDFObjectLocator(
    ThreadedComponent[TSDFObjectLocatorInputs, TSDFObjectLocatorOutputs]
):
    """Locates objects in 3D using TSDF depth, segmentation masks, and geometric median.

    Uses sphere-traced TSDF depth (from TSDFMapper) for temporally stable 3D
    position estimates. Optionally gates with live per-frame depth to reject
    stale TSDF regions. Applies the Weiszfeld geometric median for robust
    outlier-resistant 3D centre estimation.
    """

    description = "Locates objects in 3D using TSDF depth, segmentation masks, and geometric median"
    tags = Tag(io={"conduit"}, functionality={"image"})

    def __init__(self, config: TSDFObjectLocatorConfig) -> None:
        super().__init__()
        self.config = config
        self._device = auto_device("auto")

    def run(
        self,
        inputs: TSDFObjectLocatorInputs,
        outputs: TSDFObjectLocatorOutputs,
    ) -> None:
        inputs.tsdf_depth.newest = True
        inputs.camera_params.newest = True

        if inputs.live_depth is not None:
            inputs.live_depth.newest = True
            inputs.live_depth.blocking = False

        for seg_frame in inputs.segmentations:
            if seg_frame is None:
                break

            tsdf_depth_frame = next(inputs.tsdf_depth, None)
            if tsdf_depth_frame is None:
                break

            cam_frame = next(inputs.camera_params, None)
            if cam_frame is None:
                break

            live_depth_frame = None
            if inputs.live_depth is not None:
                live_depth_frame = next(inputs.live_depth, None)

            if len(seg_frame.labels) == 0:
                outputs.locations.send(
                    ObjectLocationFrame.new(
                        labels=(),
                        positions=np.zeros((0, 3), dtype=np.float32),
                        depths=np.zeros((0,), dtype=np.float32),
                        scores=np.zeros((0,), dtype=np.float32),
                        boxes=np.zeros((0, 4), dtype=np.float32),
                        object_ids=np.zeros((0,), dtype=np.int64),
                    )
                )
                continue

            locations = self._locate_objects(
                seg_frame, tsdf_depth_frame, cam_frame, live_depth_frame
            )
            outputs.locations.send(locations)

    def _locate_objects(
        self,
        seg: ObjectSegmentationFrame,
        tsdf_depth: DepthFrame,
        cam: CameraParamsFrame,
        live_depth: DepthFrame | None,
    ) -> ObjectLocationFrame:
        cfg = self.config
        device = self._device

        D_map = tsdf_depth.data  # (Hd, Wd) float32
        Hd, Wd = D_map.shape[:2]

        D_map_gpu = torch.as_tensor(D_map, device=device)
        D_live_gpu = None
        if cfg.gate_with_live_depth and live_depth is not None:
            D_live_gpu = torch.as_tensor(live_depth.data, device=device)

        # Scale intrinsics from source resolution to depth map resolution.
        K = cam.intrinsics.astype(np.float64)
        scale_x = Wd / cam.width
        scale_y = Hd / cam.height
        fx = K[0, 0] * scale_x
        fy = K[1, 1] * scale_y
        cx = K[0, 2] * scale_x
        cy = K[1, 2] * scale_y

        # Extrinsics: world-to-camera → camera-to-world.
        w2c = cam.extrinsics.astype(np.float64)
        c2w = np.linalg.inv(w2c)
        R = torch.as_tensor(c2w[:3, :3], dtype=torch.float64, device=device)
        t = torch.as_tensor(c2w[:3, 3], dtype=torch.float64, device=device)

        K_count = len(seg.labels)
        positions = np.zeros((K_count, 3), dtype=np.float32)
        median_depths = np.zeros((K_count,), dtype=np.float32)

        for i in range(K_count):
            mask_i = seg.masks[i]  # (Hs, Ws) bool
            if mask_i.shape != (Hd, Wd):
                mask_i = cv2.resize(
                    mask_i.astype(np.uint8),
                    (Wd, Hd),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            mask_gpu = torch.as_tensor(mask_i, device=device)

            valid = (D_map_gpu > 0) & mask_gpu

            # Optional live-depth gating.
            if D_live_gpu is not None:
                live_ok = D_live_gpu > 0
                agree = torch.abs(D_map_gpu - D_live_gpu) < cfg.depth_tolerance
                valid = valid & (agree | ~live_ok)

            n_valid = valid.sum().item()
            if n_valid < cfg.min_pixels:
                positions[i] = [np.nan, np.nan, np.nan]
                median_depths[i] = np.nan
                continue

            # Deproject valid pixels to camera-space 3D.
            vs, us = torch.where(valid)
            z = D_map_gpu[vs, us].double()
            x = (us.double() - cx) * z / fx
            y = (vs.double() - cy) * z / fy
            pts_cam = torch.stack([x, y, z], dim=-1)

            # Transform to world frame.
            pts_world = (R @ pts_cam.T).T + t

            # Geometric median for robust 3D centre.
            center = _geometric_median(pts_world, cfg.weiszfeld_iters)
            positions[i] = center.cpu().float().numpy()
            median_depths[i] = float(z.median().item())

        return ObjectLocationFrame.new(
            labels=seg.labels,
            positions=positions,
            depths=median_depths,
            scores=seg.scores.copy(),
            boxes=seg.boxes.copy(),
            object_ids=seg.object_ids.copy(),
        )
