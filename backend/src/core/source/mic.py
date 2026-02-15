from __future__ import annotations

from typing import TypedDict

import sounddevice as sd

from src.core.component import Component
from src.core.channel import Channel


from src.core.config import BaseConfig


from src.core.frames import AudioFrame


class MicConfig(BaseConfig):
    sample_rate: int = 48000
    channels: int = 1
    frame_ms: int = 20


class MicOutputs(TypedDict):
    audio: Channel[AudioFrame]


class Mic(Component[[], MicOutputs]):

    def __init__(self, config: MicConfig | None = None) -> None:
        super().__init__(config or MicConfig())
        self.config: MicConfig  # Type hint for IDE
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels
        self._frame_samples = int(self._sample_rate * self.config.frame_ms / 1000)
        self._output_audio: Channel[AudioFrame] = Channel(name="audio")

    def get_output_channels(self) -> MicOutputs:
        return {"audio": self._output_audio}

    def run(self) -> None:
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
        ) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(self._frame_samples)
                frame = AudioFrame(
                    data=data,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                )
                self._output_audio.send(frame)
