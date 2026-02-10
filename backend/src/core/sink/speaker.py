from __future__ import annotations

from typing import Never

import numpy as np
import sounddevice as sd

from ..component import Component
from ..topic import NOTOPIC, Topic


class Speaker(Component[bytes, Never]):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels

    def get_output_topics(self) -> tuple[Topic[Never], Topic[Never], Topic[Never], Topic[Never]]:
        return (NOTOPIC, NOTOPIC, NOTOPIC, NOTOPIC)

    def set_input_topics(self, t1: Topic[bytes], t2: Topic[Never] = NOTOPIC, t3: Topic[Never] = NOTOPIC, t4: Topic[Never] = NOTOPIC) -> None:
        self._input_topic = t1

    def run(self) -> None:
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for pcm in self._input_topic.stream(self.stop_event):
                if pcm is None:
                    break
                data = np.frombuffer(pcm, dtype=np.int16)
                if self._channels == 1:
                    data = data.reshape(-1, 1)
                else:
                    data = data.reshape(-1, self._channels)
                stream.write(data)
