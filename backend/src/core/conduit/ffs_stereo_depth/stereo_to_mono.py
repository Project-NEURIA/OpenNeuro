"""StereoToMonocularVideo conduit — extract a single eye from a stereo frame."""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import StereoVideoFrame, VideoFrame


class StereoToMonocularVideoConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"configOptions": {"eye": {}}})

    eye: Literal["left", "right"] = "left"


class StereoToMonocularVideoInputs(NamedTuple):
    stereo_video: Receiver[StereoVideoFrame]


class StereoToMonocularVideoOutputs(NamedTuple):
    video: Sender[VideoFrame]


class StereoToMonocularVideo(
    Component[StereoToMonocularVideoInputs, StereoToMonocularVideoOutputs]
):
    """Extracts a single eye (left or right) from a StereoVideoFrame."""

    def __init__(self, config: StereoToMonocularVideoConfig) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def get_config_options(
        cls, field: str, values: dict[str, Any] | None = None
    ) -> list[dict[str, str]] | None:
        if field != "config.eye":
            return None
        return [
            {"value": "left", "label": "Left Eye"},
            {"value": "right", "label": "Right Eye"},
        ]

    def run(
        self,
        inputs: StereoToMonocularVideoInputs,
        outputs: StereoToMonocularVideoOutputs,
    ) -> None:
        eye = self.config.eye

        print(f"[StereoToMonocularVideo] Extracting {eye} eye")
        for stereo_frame in inputs.stereo_video(self):
            if stereo_frame is None:
                break

            data = stereo_frame.left if eye == "left" else stereo_frame.right
            outputs.video.send(VideoFrame.new(data=data, format=stereo_frame.format))

        print("[StereoToMonocularVideo] Stopped")
