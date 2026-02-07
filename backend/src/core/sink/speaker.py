from __future__ import annotations

import numpy as np
import sounddevice as sd

from ..node import Node
from ..topic import Stream


class Speaker(Node):
    def __init__(
        self,
        input_stream: Stream[bytes],
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        self._input_stream = input_stream
        self._sample_rate = sample_rate
        self._channels = channels
        super().__init__()

    def run(self) -> None:
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for pcm in self._input_stream:
                data = np.frombuffer(pcm, dtype=np.int16)
                if self._channels == 1:
                    data = data.reshape(-1, 1)
                else:
                    data = data.reshape(-1, self._channels)
                stream.write(data)
