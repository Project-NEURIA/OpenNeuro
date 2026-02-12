from __future__ import annotations

from typing import TypedDict

import numpy as np
import sounddevice as sd

from src.core.component import Component
from src.core.channel import Channel


class SpeakerOutputs(TypedDict):
    pass


class Speaker(Component[[Channel[bytes]], SpeakerOutputs]):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels

    def output_channels(self) -> SpeakerOutputs:
        return {}

    def run(self, audio: Channel[bytes]) -> None:
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for pcm in audio.stream(self.stop_event):
                if pcm is None:
                    break
                data = np.frombuffer(pcm, dtype=np.int16)
                if self._channels == 1:
                    data = data.reshape(-1, 1)
                else:
                    data = data.reshape(-1, self._channels)
                stream.write(data)
