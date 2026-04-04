from __future__ import annotations

import cv2

from src.lib.vision.video_source import VideoSource, VideoSourceConfig


class VideoPlayerConfig(VideoSourceConfig):
    source: str = ""
    loop: bool = True


class VideoPlayer(VideoSource):
    description = "Plays back video from a file"

    """Plays a local video file and sends VideoFrame through the pipeline."""

    _config: VideoPlayerConfig

    def __init__(self, config: VideoPlayerConfig = VideoPlayerConfig()) -> None:
        super().__init__(config)

    def _on_eof(self, cap: cv2.VideoCapture) -> None:
        if self._config.loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
