from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from pydantic import BaseModel

from src.core.channel import Sender, UIVideoReceiver
from src.core.component import ThreadedComponent, Tag
from src.core.frames import VideoDataFormat, VideoFrame


class CameraBrowserConfig(BaseModel):
    pass


class CameraBrowserInputs(NamedTuple):
    ui_video: UIVideoReceiver


class CameraBrowserOutputs(NamedTuple):
    video: Sender[VideoFrame]


class CameraBrowser(ThreadedComponent[CameraBrowserInputs, CameraBrowserOutputs]):
    tags = Tag(io={"source"}, functionality={"vision"})
    description = "Captures live **video** from a **browser camera**. Receives JPEG frames via the UI channel and streams `VideoFrame` data."

    def __init__(self, config: CameraBrowserConfig = CameraBrowserConfig()) -> None:
        super().__init__()
        self.config: CameraBrowserConfig = config

    def run(self, inputs: CameraBrowserInputs, outputs: CameraBrowserOutputs) -> None:
        for chunk in inputs.ui_video:
            if chunk is None:
                break

            # chunk is expected to be JPEG bytes coming from the browser
            np_arr = np.frombuffer(chunk, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                frame = VideoFrame.new(data=img, format=VideoDataFormat.BGR)
                outputs.video.send(frame)
