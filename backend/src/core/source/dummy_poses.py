"""DummyPosesInput source — emits a T-pose that sways back-and-forth with head turning."""

from __future__ import annotations

import math
import time
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Sender
from src.core.component import PrimitiveComponent
from src.core.frames import BodyPoseFrame, BonePose


# Default T-pose positions (meters)
_TPOSE: dict[str, BonePose] = {
    "head": BonePose(pos_x=0.0, pos_y=1.70, pos_z=0.0),
    "waist": BonePose(pos_x=0.0, pos_y=0.93, pos_z=0.0),
    "chest": BonePose(pos_x=0.0, pos_y=1.29, pos_z=0.0),
    "left_shoulder": BonePose(pos_x=-0.15, pos_y=1.41, pos_z=0.0),
    "right_shoulder": BonePose(pos_x=0.15, pos_y=1.41, pos_z=0.0),
    "left_elbow": BonePose(pos_x=-0.45, pos_y=1.41, pos_z=0.0),
    "right_elbow": BonePose(pos_x=0.45, pos_y=1.41, pos_z=0.0),
    "left_hand": BonePose(pos_x=-0.67, pos_y=1.41, pos_z=0.0),
    "right_hand": BonePose(pos_x=0.67, pos_y=1.41, pos_z=0.0),
    "left_knee": BonePose(pos_x=-0.09, pos_y=0.46, pos_z=0.0),
    "right_knee": BonePose(pos_x=0.09, pos_y=0.46, pos_z=0.0),
    "left_foot": BonePose(pos_x=-0.09, pos_y=0.06, pos_z=0.0),
    "right_foot": BonePose(pos_x=0.09, pos_y=0.06, pos_z=0.0),
}


class DummyPosesConfig(BaseModel):
    center_x: float = 0.0
    center_z: float = 0.0
    body_frequency: float = 0.3
    body_amplitude: float = 0.5
    head_frequency: float = 0.5
    head_amplitude: float = 0.6
    fps: float = 60.0


class DummyPosesOutputs(NamedTuple):
    poses: Sender[BodyPoseFrame]


class DummyPosesInput(PrimitiveComponent[tuple[()], DummyPosesOutputs]):
    """Source that emits a T-pose swaying back-and-forth along Z with the head turning left/right."""

    def __init__(self, config: DummyPosesConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, inputs: tuple[()], outputs: DummyPosesOutputs) -> None:
        cfg = self.config
        interval = 1.0 / cfg.fps
        t0 = time.monotonic()
        print(
            f"[DummyPosesInput] Running at {cfg.fps} fps "
            f"(body {cfg.body_frequency} Hz, head {cfg.head_frequency} Hz)"
        )

        while not self.stop_event.is_set():
            t = time.monotonic() - t0

            # Body sways back-and-forth along Z axis
            body_z = math.sin(2 * math.pi * cfg.body_frequency * t) * cfg.body_amplitude

            # Head yaw (rotation around Y axis) — quaternion for Y-axis rotation
            head_angle = (
                math.sin(2 * math.pi * cfg.head_frequency * t) * cfg.head_amplitude
            )
            half = head_angle / 2.0
            head_rot_w = math.cos(half)
            head_rot_y = math.sin(half)

            poses: dict[str, BonePose | None] = {}
            for name, base in _TPOSE.items():
                if name == "head":
                    poses[name] = BonePose(
                        pos_x=base.pos_x + cfg.center_x,
                        pos_y=base.pos_y,
                        pos_z=base.pos_z + body_z + cfg.center_z,
                        rot_w=head_rot_w,
                        rot_x=0.0,
                        rot_y=head_rot_y,
                        rot_z=0.0,
                    )
                else:
                    poses[name] = BonePose(
                        pos_x=base.pos_x + cfg.center_x,
                        pos_y=base.pos_y,
                        pos_z=base.pos_z + body_z + cfg.center_z,
                    )

            outputs.poses.send(BodyPoseFrame(poses=poses))
            self.stop_event.wait(interval)

        print("[DummyPosesInput] Stopped")
