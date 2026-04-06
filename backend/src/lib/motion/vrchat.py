"""VRChatVideo source — stereo video capture from VRChat via OpenVR virtual driver."""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel

from src.core.channel import Sender
from src.core.component import ThreadedComponent, Tag
from src.core.config import PCVR_IP
from src.core.frames import (
    StereoCameraParamsFrame,
    StereoVideoFrame,
    VideoDataFormat,
)


class VRChatVideoConfig(BaseModel):
    host: str = PCVR_IP
    port: int = 21213
    fps: int = 30


class VRChatVideoOutputs(NamedTuple):
    stereo_video: Sender[StereoVideoFrame]
    camera_params: Sender[StereoCameraParamsFrame]


class VRChatVideo(ThreadedComponent[tuple[()], VRChatVideoOutputs]):
    description = "Captures stereo video from VRChat"

    """Captures stereo video frames from VRChat via the OpenVR virtual driver.

    Streams synchronized left+right eye pairs at the configured FPS.
    """

    tags = Tag(io={"source"}, functionality={"motion"})

    def __init__(self, config: VRChatVideoConfig = VRChatVideoConfig()) -> None:
        super().__init__()
        self._host = config.host
        self._port = config.port
        self._frame_interval = 1.0 / config.fps

    def run(self, inputs: tuple[()], outputs: VRChatVideoOutputs) -> None:
        from ovd_client import Client

        with Client(host=self._host, port=self._port) as client:
            last_send = 0.0

            with client.frame_stream() as frames:
                intrinsics = client.get_intrinsics()
                baseline = client.get_ipd()

                for frame in frames:
                    if self.stop_event.is_set():
                        break

                    now = time.monotonic()
                    if now - last_send < self._frame_interval:
                        continue
                    last_send = now

                    left_rgba = np.frombuffer(frame.left, dtype=np.uint8).reshape(
                        frame.height, frame.width, 4
                    )
                    left_rgb = left_rgba[:, :, :3].copy()

                    right_rgba = np.frombuffer(frame.right, dtype=np.uint8).reshape(
                        frame.height, frame.width, 4
                    )
                    right_rgb = right_rgba[:, :, :3].copy()

                    outputs.stereo_video.send(
                        StereoVideoFrame.new(
                            left=left_rgb,
                            right=right_rgb,
                            format=VideoDataFormat.RGB,
                        )
                    )
                    # Convert extrinsics from OpenVR (right-handed, -Z forward)
                    # to our convention (left-handed, +Z forward) by negating Z.
                    ext = client.get_extrinsics(frame, eye=0)
                    ext[2, :] = -ext[2, :]  # flip Z row
                    ext[:, 2] = -ext[:, 2]  # flip Z column

                    outputs.camera_params.send(
                        StereoCameraParamsFrame.new(
                            intrinsics=intrinsics,
                            extrinsics=ext,
                            baseline=baseline,
                            width=frame.width,
                            height=frame.height,
                        )
                    )


import os  # noqa: E402

if os.environ.get("DEPLOY_MODE") == "remote":
    VRChatVideo._registerable = False
