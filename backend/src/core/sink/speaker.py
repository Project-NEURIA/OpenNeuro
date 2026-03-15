from __future__ import annotations

from typing import NamedTuple

import numpy as np
import sounddevice as sd
from pydantic import BaseModel

from src.core.channel import Receiver
from src.core.component import PrimitiveComponent, Tag
from src.core.frames import AudioDataFormat, AudioFrame


class SpeakerConfig(BaseModel):
    sample_rate: int = 24000
    channels: int = 1


class SpeakerInputs(NamedTuple):
    audio: Receiver[AudioFrame]


class Speaker(PrimitiveComponent[SpeakerInputs, tuple[()]]):
    _tags = Tag(io={"sink"}, functionality={"audio"})

    def __init__(self, config: SpeakerConfig) -> None:
        super().__init__()
        self.config: SpeakerConfig = config
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels

    def run(self, inputs: SpeakerInputs, outputs: tuple[()]) -> None:
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for frame in inputs.audio(self):
                if frame is None:
                    break

                pcm_bytes = frame.get(
                    sample_rate=self._sample_rate,
                    num_channels=self._channels,
                    data_format=AudioDataFormat.PCM16,
                )

                stream.write(
                    np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, self._channels)
                )
