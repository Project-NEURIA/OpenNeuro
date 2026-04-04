"""ObjectLocator conduit — 3D localisation of segmented objects via depth deprojection."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import (
    CameraParamsFrame,
    DepthFrame,
    ObjectLocationFrame,
    ObjectSegmentationFrame,
)


class ObjectLocatorInputs(NamedTuple):
    segmentations: Receiver[ObjectSegmentationFrame]
    depth: Receiver[DepthFrame]
    camera_params: Receiver[CameraParamsFrame]


class ObjectLocatorOutputs(NamedTuple):
    locations: Sender[ObjectLocationFrame]


class ObjectLocator(ThreadedComponent[ObjectLocatorInputs, ObjectLocatorOutputs]):
    description = "Computes 3D world positions for segmented objects"
    tags = Tag(io={"conduit"}, functionality={"vision"})

    """Computes 3D world positions for segmented objects using depth deprojection.

    For each segmentation mask, applies it to the depth map, deprojects
    masked pixels to 3D camera-space coordinates, takes coordinate-wise
    median as the object position, then transforms to world frame.
    """

    def run(self, inputs: ObjectLocatorInputs, outputs: ObjectLocatorOutputs) -> None:
        inputs.depth.newest = True
        inputs.camera_params.newest = True

        for seg_frame in inputs.segmentations:
            if seg_frame is None:
                break

            depth_frame = next(inputs.depth, None)
            if depth_frame is None:
                break

            cam_frame = next(inputs.camera_params, None)
            if cam_frame is None:
                break

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

            locations = _locate_objects(seg_frame, depth_frame, cam_frame)
            outputs.locations.send(locations)


def _locate_objects(
    seg: ObjectSegmentationFrame,
    depth: DepthFrame,
    cam: CameraParamsFrame,
) -> ObjectLocationFrame:
    depth_data = depth.data  # (Hd, Wd) float32
    Hd, Wd = depth_data.shape[:2]

    # Scale intrinsics from source resolution to depth map resolution.
    K = cam.intrinsics.astype(np.float64)
    scale_x = Wd / cam.width
    scale_y = Hd / cam.height
    fx = K[0, 0] * scale_x
    fy = K[1, 1] * scale_y
    cx = K[0, 2] * scale_x
    cy = K[1, 2] * scale_y

    # Extrinsics are world-to-camera; invert to get camera-to-world.
    w2c = cam.extrinsics.astype(np.float64)
    c2w = np.linalg.inv(w2c)
    R = c2w[:3, :3]
    t = c2w[:3, 3]

    # Pre-build pixel coordinate grids for the depth map.
    u_grid, v_grid = np.meshgrid(
        np.arange(Wd, dtype=np.float64),
        np.arange(Hd, dtype=np.float64),
    )

    K_count = len(seg.labels)
    positions = np.zeros((K_count, 3), dtype=np.float32)
    median_depths = np.zeros((K_count,), dtype=np.float32)

    for i in range(K_count):
        mask_i = seg.masks[i]  # (Hs, Ws) bool
        # Resize mask to depth resolution if needed.
        if mask_i.shape != (Hd, Wd):
            mask_i = cv2.resize(
                mask_i.astype(np.uint8),
                (Wd, Hd),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # Get depth values under the mask.
        d = depth_data[mask_i].astype(np.float64)
        valid = np.isfinite(d) & (d > 0)
        d = d[valid]

        if len(d) == 0:
            positions[i] = [np.nan, np.nan, np.nan]
            median_depths[i] = np.nan
            continue

        # Pixel coordinates of valid masked depth pixels.
        u = u_grid[mask_i][valid]
        v = v_grid[mask_i][valid]

        # Deproject to camera space (pinhole model).
        X_cam = (u - cx) * d / fx
        Y_cam = (v - cy) * d / fy
        Z_cam = d

        # Coordinate-wise median in camera space.
        cam_pos = np.array(
            [np.median(X_cam), np.median(Y_cam), np.median(Z_cam)],
            dtype=np.float64,
        )

        # Transform to world frame.
        world_pos = cam_pos @ R.T + t
        positions[i] = world_pos.astype(np.float32)
        median_depths[i] = float(np.median(Z_cam))

    return ObjectLocationFrame.new(
        labels=seg.labels,
        positions=positions,
        depths=median_depths,
        scores=seg.scores.copy(),
        boxes=seg.boxes.copy(),
        object_ids=seg.object_ids.copy(),
    )
