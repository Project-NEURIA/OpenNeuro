from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Sender, UIAudioReceiver
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioFrame


class MicBrowserConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    # Note: we generally keep these settings in sync with what the browser captures


class MicBrowserInputs(NamedTuple):
    ui_audio: UIAudioReceiver


class MicBrowserOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class MicBrowser(ThreadedComponent[MicBrowserInputs, MicBrowserOutputs]):
    tags = Tag(io={"source"}, functionality={"audio"})
    description = "Captures audio from a **browser microphone**. Receives raw audio data via the UI channel and streams `AudioFrame` data at a configurable *sample rate* and *channel count*."

    def __init__(self, config: MicBrowserConfig = MicBrowserConfig()) -> None:
        super().__init__()
        self.config: MicBrowserConfig = config
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels

    def run(self, inputs: MicBrowserInputs, outputs: MicBrowserOutputs) -> None:
        for chunk in inputs.ui_audio:
            if chunk is None:
                break

            # chunk is expected to be bytes of INT16 audio coming from the browser
            frame = AudioFrame.new(
                data=chunk,
                sample_rate=self._sample_rate,
                channels=self._channels,
            )
            outputs.audio.send(frame)
