"""StereoCameraParamsAdapter — converts StereoCameraParamsFrame to CameraParamsFrame."""

from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent
from src.core.frames import CameraParamsFrame, StereoCameraParamsFrame


class StereoCameraParamsAdapterInputs(NamedTuple):
    stereo_camera_params: Receiver[StereoCameraParamsFrame]


class StereoCameraParamsAdapterOutputs(NamedTuple):
    camera_params: Sender[CameraParamsFrame]


class StereoCameraParamsAdapter(
    PrimitiveComponent[
        StereoCameraParamsAdapterInputs, StereoCameraParamsAdapterOutputs
    ]
):
    """Strips baseline from StereoCameraParamsFrame, emitting CameraParamsFrame."""

    def run(
        self,
        inputs: StereoCameraParamsAdapterInputs,
        outputs: StereoCameraParamsAdapterOutputs,
    ) -> None:
        for frame in inputs.stereo_camera_params(self, newest=True):
            if frame is None:
                break
            outputs.camera_params.send(
                CameraParamsFrame.new(
                    intrinsics=frame.intrinsics,
                    extrinsics=frame.extrinsics,
                    width=frame.width,
                    height=frame.height,
                )
            )
