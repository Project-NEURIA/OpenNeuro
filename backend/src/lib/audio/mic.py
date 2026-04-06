from __future__ import annotations

from typing import NamedTuple

import sounddevice as sd
from pydantic import BaseModel

from src.core.channel import Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioFrame


class MicConfig(BaseModel):
    sample_rate: int = 48000
    channels: int = 1
    frame_ms: int = 20


class MicOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class Mic(ThreadedComponent[tuple[()], MicOutputs]):
    tags = Tag(io={"source"}, functionality={"audio"})
    description = "Captures audio from a system **microphone**. Streams raw `AudioFrame` data at a configurable *sample rate*, *channel count*, and *frame duration*."

    def __init__(self, config: MicConfig) -> None:
        super().__init__()
        self.config: MicConfig = config
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels
        self._frame_samples = int(self._sample_rate * self.config.frame_ms / 1000)

    def run(self, inputs: tuple[()], outputs: MicOutputs) -> None:
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
        ) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(self._frame_samples)
                frame = AudioFrame.new(
                    data=data,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                )
                outputs.audio.send(frame)


import os  # noqa: E402

if os.environ.get("DEPLOY_MODE") == "remote":
    Mic._registerable = False
