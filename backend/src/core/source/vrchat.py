from __future__ import annotations

import io
import time
from typing import TypedDict

import numpy as np
from PIL import Image

from src.core.component import Component
from src.core.channel import Channel


class VRChatVideoOutputs(TypedDict):
    video: Channel[bytes]


class VRChatVideo(Component[[], VRChatVideoOutputs]):
    """Captures video frames from VRChat via the OpenVR virtual driver."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6969,
        fps: int = 30,
        quality: int = 75,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._frame_interval = 1.0 / fps
        self._quality = quality
        self._output_video: Channel[bytes] = Channel(name="video")

    def get_output_channels(self) -> VRChatVideoOutputs:
        return {"video": self._output_video}

    def run(self) -> None:
        from ovd_client import Client

        with Client() as client:
            client.connect(host=self._host, port=self._port)
            last_send = 0.0

            for frame in client.frame_stream():
                if self.stop_event.is_set():
                    break

                now = time.monotonic()
                if now - last_send < self._frame_interval:
                    continue
                last_send = now

                rgba = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                    frame.height, frame.width, 4
                )
                img = Image.fromarray(rgba[:, :, :3])

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self._quality)
                self._output_video.send(buf.getvalue())
