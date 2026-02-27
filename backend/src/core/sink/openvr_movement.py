"""OpenVR movement sink — sends body poses to SteamVR via the virtual driver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from ovd_client import Client, Pose
from pydantic import BaseModel

from src.core.component import Component
from src.core.channel import Receiver
from src.core.frames import BodyPoseFrame, BonePose


# Default T-pose body positions (heights in meters, identity rotation)
_TPOSE: dict[str, Pose] = {
    "head": Pose(
        pos_x=0.0, pos_y=1.70, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "waist": Pose(
        pos_x=0.0, pos_y=0.93, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "chest": Pose(
        pos_x=0.0, pos_y=1.29, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "left_shoulder": Pose(
        pos_x=-0.15, pos_y=1.41, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "right_shoulder": Pose(
        pos_x=0.15, pos_y=1.41, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "left_elbow": Pose(
        pos_x=-0.45, pos_y=1.41, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "right_elbow": Pose(
        pos_x=0.45, pos_y=1.41, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "left_hand": Pose(
        pos_x=-0.67, pos_y=1.41, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "right_hand": Pose(
        pos_x=0.67, pos_y=1.41, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "left_knee": Pose(
        pos_x=-0.09, pos_y=0.46, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "right_knee": Pose(
        pos_x=0.09, pos_y=0.46, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "left_foot": Pose(
        pos_x=-0.09, pos_y=0.06, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
    "right_foot": Pose(
        pos_x=0.09, pos_y=0.06, pos_z=0.0, rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0
    ),
}


class OpenVRMovementConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 21213


class OpenVRMovementInputs(NamedTuple):
    poses: Receiver[BodyPoseFrame] | None


class OpenVRMovementOutputs(NamedTuple):
    pass


def _to_ovd_pose(bone: BonePose | None) -> Pose | None:
    """Convert a BonePose to an ovd_client Pose, or None."""
    if bone is None:
        return None
    return Pose(
        pos_x=bone.pos_x,
        pos_y=bone.pos_y,
        pos_z=bone.pos_z,
        rot_w=bone.rot_w,
        rot_x=bone.rot_x,
        rot_y=bone.rot_y,
        rot_z=bone.rot_z,
    )


def _send_poses(client: Client, poses: Mapping[str, Pose | None]) -> None:
    """Send a dict of body part poses to the driver."""
    client.update_pose(
        head=poses.get("head"),
        left_hand=poses.get("left_hand"),
        right_hand=poses.get("right_hand"),
        waist=poses.get("waist"),
        chest=poses.get("chest"),
        left_foot=poses.get("left_foot"),
        right_foot=poses.get("right_foot"),
        left_knee=poses.get("left_knee"),
        right_knee=poses.get("right_knee"),
        left_elbow=poses.get("left_elbow"),
        right_elbow=poses.get("right_elbow"),
        left_shoulder=poses.get("left_shoulder"),
        right_shoulder=poses.get("right_shoulder"),
    )


class OpenVRMovementSink(
    Component[OpenVRMovementInputs, OpenVRMovementOutputs]  # type: ignore[type-var]
):
    """Sink that streams body pose frames to the OpenVR virtual driver.

    If no input channel is provided, sends a static T-pose.
    """

    def __init__(self, config: OpenVRMovementConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, inputs: OpenVRMovementInputs, outputs: OpenVRMovementOutputs) -> None:
        client = Client(host=self.config.host, port=self.config.port)
        client.connect()
        print(
            f"[OpenVRMovement] Connected to driver at {self.config.host}:{self.config.port}"
        )

        try:
            poses = inputs.poses
            if poses is None:
                print("[OpenVRMovement] No input channel, sending T-pose")
                _send_poses(client, _TPOSE)
                self.stop_event.wait()
            else:
                for frame in poses(self):
                    if frame is None:
                        break

                    p = frame.get()
                    _send_poses(client, {k: _to_ovd_pose(v) for k, v in p.items()})
        finally:
            client.disconnect()
            print("[OpenVRMovement] Disconnected from driver")
