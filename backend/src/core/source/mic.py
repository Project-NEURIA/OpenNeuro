from __future__ import annotations

import sounddevice as sd

from ..node import Node
from ..topic import Topic


class Mic(Node[bytes]):
    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        channels: int = 1,
        frame_ms: int = 20,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_samples = int(sample_rate * frame_ms / 1000)
        super().__init__(Topic[bytes]())

    def run(self) -> None:
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
        ) as stream:
            while not self._stopped:
                data, _ = stream.read(self._frame_samples)
                self.topic.send(data.tobytes())
