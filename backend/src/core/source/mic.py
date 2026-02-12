from __future__ import annotations

from typing import TypedDict

import sounddevice as sd

from src.core.component import Component
from src.core.channel import Channel


class MicOutputs(TypedDict):
    audio: Channel[bytes]


class Mic(Component[[], MicOutputs]):

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        channels: int = 1,
        frame_ms: int = 20,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_samples = int(sample_rate * frame_ms / 1000)
        self._output_audio: Channel[bytes] = Channel(name="audio")

    def output_channels(self) -> MicOutputs:
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
                self._output_audio.send(data.tobytes())
