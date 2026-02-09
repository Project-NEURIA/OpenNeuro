from __future__ import annotations

from typing import Never

import numpy as np
import sounddevice as sd

from ..node import Node
from ..topic import Topic


class Speaker(Node[Never]):
    def __init__(
        self,
        input_topic: Topic[bytes],
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._input_topic = input_topic
        super().__init__()

    def set_input_topics(self, *topics: Topic) -> None:
        self._input_topic = topics[0]

    def run(self) -> None:
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for pcm in self._input_topic.stream(self.stop_event):
                if self._stopped:
                    break
                data = np.frombuffer(pcm, dtype=np.int16)
                if self._channels == 1:
                    data = data.reshape(-1, 1)
                else:
                    data = data.reshape(-1, self._channels)
                stream.write(data)
