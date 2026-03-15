from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from src.core.channel import Sender
from src.core.component import PrimitiveComponent
from src.core.frames import CameraParamsFrame, VideoDataFormat, VideoFrame


class VRChatVideoOutputs(NamedTuple):
    video: Sender[VideoFrame]
    camera_params: Sender[CameraParamsFrame]


class VRChatVideo(PrimitiveComponent[tuple[()], VRChatVideoOutputs]):
    """Captures video frames from VRChat via the OpenVR virtual driver.

    Streams left-eye frames only (monocular) at the configured FPS.
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

    def run(self, inputs: tuple[()], outputs: VRChatVideoOutputs) -> None:
        from ovd_client import Client

        with Client(host=self._host, port=self._port) as client:
            last_send = 0.0
            intrinsics = client.get_intrinsics()

            with client.frame_stream() as frames:
                for frame in frames:
                    if self.stop_event.is_set():
                        break

                    if frame.eye != 0:
                        continue

                    now = time.monotonic()
                    if now - last_send < self._frame_interval:
                        continue
                    last_send = now

                    rgba = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                        frame.height, frame.width, 4
                    )
                    rgb = rgba[:, :, :3]  # RGBA -> RGB (drop alpha)

                    outputs.video.send(
                        VideoFrame.new(data=rgb, format=VideoDataFormat.RGB)
                    )
                    outputs.camera_params.send(
                        CameraParamsFrame.new(
                            intrinsics=intrinsics,
                            extrinsics=client.get_extrinsics(frame),
                            width=frame.width,
                            height=frame.height,
                        )
                    )
