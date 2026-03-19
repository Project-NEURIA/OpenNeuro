from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import ThreadedComponent, Tag
from src.core.frames import VideoDataFormat, VideoFrame
from src.core.channel import UIVideoSender


class VideoStreamInputs(NamedTuple):
    video: Receiver[VideoFrame]


class VideoStreamOutputs(NamedTuple):
    ui_video: UIVideoSender


class VideoStream(ThreadedComponent[VideoStreamInputs, VideoStreamOutputs]):
    description = "Streams video frames to the UI"

    """Receives video frames and streams JPEG to the frontend via UI channel."""

    tags = Tag(io={"sink"}, functionality={"video"})

    def run(self, inputs: VideoStreamInputs, outputs: VideoStreamOutputs) -> None:
        import cv2

        for frame in inputs.video(self, newest=True):
            if frame is None:
                break
            _, buf = cv2.imencode(
                ".jpg",
                frame.get(VideoDataFormat.BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            outputs.ui_video.send(buf.tobytes())
