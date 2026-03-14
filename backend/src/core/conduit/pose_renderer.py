"""PoseRenderer conduit — renders BodyPoseFrame skeletons to JPEG bytes using OpenCV."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent
from src.core.frames import BodyPoseFrame, BonePose, VideoFrame, VideoDataFormat

# Skeleton connectivity: (parent, child) pairs
_BONES: list[tuple[str, str]] = [
    ("chest", "head"),
    ("waist", "chest"),
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("waist", "left_knee"),
    ("left_knee", "left_foot"),
    ("waist", "right_knee"),
    ("right_knee", "right_foot"),
]

_BG_COLOR = (30, 30, 30)  # dark gray
_JOINT_RADIUS_PX = 5
_BONE_THICKNESS = 2

# Per-limb colors (BGR)
_HEAD_COLOR = (0, 200, 255)  # yellow
_TORSO_COLOR = (200, 200, 200)  # white-ish
_LEFT_ARM_COLOR = (0, 180, 0)  # green
_RIGHT_ARM_COLOR = (255, 100, 0)  # blue
_LEFT_LEG_COLOR = (0, 255, 255)  # yellow-green
_RIGHT_LEG_COLOR = (255, 0, 200)  # magenta

_BONE_COLORS: dict[tuple[str, str], tuple[int, int, int]] = {
    ("chest", "head"): _HEAD_COLOR,
    ("waist", "chest"): _TORSO_COLOR,
    ("chest", "left_shoulder"): _LEFT_ARM_COLOR,
    ("left_shoulder", "left_elbow"): _LEFT_ARM_COLOR,
    ("left_elbow", "left_hand"): _LEFT_ARM_COLOR,
    ("chest", "right_shoulder"): _RIGHT_ARM_COLOR,
    ("right_shoulder", "right_elbow"): _RIGHT_ARM_COLOR,
    ("right_elbow", "right_hand"): _RIGHT_ARM_COLOR,
    ("waist", "left_knee"): _LEFT_LEG_COLOR,
    ("left_knee", "left_foot"): _LEFT_LEG_COLOR,
    ("waist", "right_knee"): _RIGHT_LEG_COLOR,
    ("right_knee", "right_foot"): _RIGHT_LEG_COLOR,
}

_JOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "head": _HEAD_COLOR,
    "chest": _TORSO_COLOR,
    "waist": _TORSO_COLOR,
    "left_shoulder": _LEFT_ARM_COLOR,
    "left_elbow": _LEFT_ARM_COLOR,
    "left_hand": _LEFT_ARM_COLOR,
    "right_shoulder": _RIGHT_ARM_COLOR,
    "right_elbow": _RIGHT_ARM_COLOR,
    "right_hand": _RIGHT_ARM_COLOR,
    "left_knee": _LEFT_LEG_COLOR,
    "left_foot": _LEFT_LEG_COLOR,
    "right_knee": _RIGHT_LEG_COLOR,
    "right_foot": _RIGHT_LEG_COLOR,
}


class PoseRendererConfig(BaseModel):
    width: int = 640
    height: int = 480


class PoseRendererInputs(NamedTuple):
    pose: Receiver[BodyPoseFrame]


class PoseRendererOutputs(NamedTuple):
    video: Sender[VideoFrame]


def _project(
    bone: BonePose,
    width: int,
    height: int,
    scale: float = 150.0,
) -> tuple[int, int]:
    """Project a 3D bone position to 2D pixel coords (front view: X right, Y up)."""
    px = int(width / 2 + bone.pos_x * scale)
    py = int(height * 0.75 - bone.pos_y * scale)  # Y is up; anchor at 75% down
    return px, py


class PoseRenderer(PrimitiveComponent[PoseRendererInputs, PoseRendererOutputs]):
    """Conduit that renders each BodyPoseFrame as a 2D skeleton and outputs JPEG bytes."""

    def __init__(self, config: PoseRendererConfig) -> None:
        super().__init__()
        self.config = config

    def _render_pose(self, poses: dict[str, BonePose | None]) -> np.ndarray:
        w, h = self.config.width, self.config.height
        img = np.full((h, w, 3), _BG_COLOR, dtype=np.uint8)

        # Draw ground plane
        ground_y = int(h * 0.75)
        img[ground_y:, :] = (40, 40, 35)  # slightly lighter/warmer than bg
        cv2.line(img, (0, ground_y), (w, ground_y), (60, 60, 50), 1, cv2.LINE_AA)

        # Project all available joints
        pts: dict[str, tuple[int, int]] = {}
        for name, bone in poses.items():
            if bone is not None:
                pts[name] = _project(bone, w, h)

        # Draw bones
        for parent, child in _BONES:
            if parent in pts and child in pts:
                color = _BONE_COLORS.get((parent, child), _TORSO_COLOR)
                cv2.line(
                    img, pts[parent], pts[child], color, _BONE_THICKNESS, cv2.LINE_AA
                )

        # Draw joints
        for name, pt in pts.items():
            color = _JOINT_COLORS.get(name, _TORSO_COLOR)
            cv2.circle(img, pt, _JOINT_RADIUS_PX, color, -1, cv2.LINE_AA)

        return img

    def run(self, inputs: PoseRendererInputs, outputs: PoseRendererOutputs) -> None:
        for frame in inputs.pose(self):
            if frame is None:
                break
            poses = frame.get()

            img = self._render_pose(poses)
            outputs.video.send(VideoFrame.new(data=img, format=VideoDataFormat.BGR))
