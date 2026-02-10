from __future__ import annotations

from typing import Never

import sounddevice as sd

from ..component import Component
from ..topic import NOTOPIC, Topic


class Mic(Component[Never, bytes]):
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
        self._output = Topic[bytes]()

    def get_output_topics(self) -> tuple[Topic[bytes], Topic[Never], Topic[Never], Topic[Never]]:
        return (self._output, NOTOPIC, NOTOPIC, NOTOPIC)

    def run(self) -> None:
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
        ) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(self._frame_samples)
                self._output.send(data.tobytes())
