from __future__ import annotations


import numpy as np
import sounddevice as sd

from ..component import Component
from ..channel import Channel


class Speaker(Component[Channel[bytes]]):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels

    def get_output_channels(self) -> tuple[()]:
        return ()

    def set_input_channels(self, t1: Channel[bytes]) -> None:
        self._input_channel = t1

    def run(self) -> None:
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for pcm in self._input_channel.stream(self.stop_event):
                if pcm is None:
                    break
                data = np.frombuffer(pcm, dtype=np.int16)
                if self._channels == 1:
                    data = data.reshape(-1, 1)
                else:
                    data = data.reshape(-1, self._channels)
                stream.write(data)
