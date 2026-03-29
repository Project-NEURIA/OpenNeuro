"""StereoCameraParamsAdapter — converts StereoCameraParamsFrame to CameraParamsFrame."""

from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import CameraParamsFrame, StereoCameraParamsFrame


class StereoCameraParamsAdapterInputs(NamedTuple):
    stereo_camera_params: Receiver[StereoCameraParamsFrame]


class StereoCameraParamsAdapterOutputs(NamedTuple):
    camera_params: Sender[CameraParamsFrame]


class StereoCameraParamsAdapter(
    ThreadedComponent[StereoCameraParamsAdapterInputs, StereoCameraParamsAdapterOutputs]
):
    description = "Converts stereo camera params to monocular camera params"
    tags = Tag(io={"conduit"}, functionality={"image"})

    """Strips baseline from StereoCameraParamsFrame, emitting CameraParamsFrame."""

    def run(
        self,
        inputs: StereoCameraParamsAdapterInputs,
        outputs: StereoCameraParamsAdapterOutputs,
    ) -> None:
        inputs.stereo_camera_params.newest = True
        for frame in inputs.stereo_camera_params:
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
