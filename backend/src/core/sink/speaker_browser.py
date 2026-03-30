from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, UIAudioSender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioDataFormat, AudioFrame


class SpeakerBrowserConfig(BaseModel):
    sample_rate: int = 24000
    channels: int = 1


class SpeakerBrowserInputs(NamedTuple):
    audio: Receiver[AudioFrame]


class SpeakerBrowserOutputs(NamedTuple):
    ui_audio: UIAudioSender


class SpeakerBrowser(ThreadedComponent[SpeakerBrowserInputs, SpeakerBrowserOutputs]):
    tags = Tag(io={"sink"}, functionality={"audio"})
    description = "Plays `AudioFrame` data through a **browser speaker**. Streams raw PCM audio over the UI channel."

    def __init__(self, config: SpeakerBrowserConfig = SpeakerBrowserConfig()) -> None:
        super().__init__()
        self.config: SpeakerBrowserConfig = config
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels

    def run(self, inputs: SpeakerBrowserInputs, outputs: SpeakerBrowserOutputs) -> None:
        for frame in inputs.audio:
            if frame is None:
                break

            pcm_bytes = frame.get(
                sample_rate=self._sample_rate,
                num_channels=self._channels,
                data_format=AudioDataFormat.PCM16,
            )

            outputs.ui_audio.send(pcm_bytes)
