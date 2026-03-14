"""StereoVRChatVideo source — stereo video capture from VRChat via OpenVR virtual driver."""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from src.core.channel import Sender
from src.core.component import PrimitiveComponent
from src.core.frames import (
    StereoCameraParamsFrame,
    StereoVideoFrame,
    VideoDataFormat,
)


class StereoVRChatVideoOutputs(NamedTuple):
    stereo_video: Sender[StereoVideoFrame]
    camera_params: Sender[StereoCameraParamsFrame]


class StereoVRChatVideo(PrimitiveComponent[tuple[()], StereoVRChatVideoOutputs]):
    """Captures stereo video frames from VRChat via the OpenVR virtual driver.

    Streams synchronized left+right eye pairs at the configured FPS.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 21213,
        fps: int = 30,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._frame_interval = 1.0 / fps

    def run(self, inputs: tuple[()], outputs: StereoVRChatVideoOutputs) -> None:
        from ovd_client import Client

        with Client(host=self._host, port=self._port) as client:
            last_send = 0.0

            with client.stereo_stream() as stereo_frames:
                intrinsics = client.get_intrinsics()
                baseline = client.get_ipd()

                for stereo in stereo_frames:
                    if self.stop_event.is_set():
                        break

                    now = time.monotonic()
                    if now - last_send < self._frame_interval:
                        continue
                    last_send = now

                    left_rgba = np.frombuffer(stereo.left.data, dtype=np.uint8).reshape(
                        stereo.left.height, stereo.left.width, 4
                    )
                    left_rgb = left_rgba[:, :, :3].copy()

                    right_rgba = np.frombuffer(
                        stereo.right.data, dtype=np.uint8
                    ).reshape(stereo.right.height, stereo.right.width, 4)
                    right_rgb = right_rgba[:, :, :3].copy()

                    outputs.stereo_video.send(
                        StereoVideoFrame.new(
                            left=left_rgb,
                            right=right_rgb,
                            format=VideoDataFormat.RGB,
                        )
                    )
                    outputs.camera_params.send(
                        StereoCameraParamsFrame.new(
                            intrinsics=intrinsics,
                            extrinsics=client.get_extrinsics(stereo.left),
                            baseline=baseline,
                        )
                    )
