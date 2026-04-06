from __future__ import annotations

from typing import Any

import sounddevice as sd
from pydantic import BaseModel, ConfigDict

from src.lib.audio.devices import (
    coerce_sounddevice_device,
    list_audio_input_devices,
)
from src.lib.audio.mic import MicOutputs
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioFrame


class AudioInConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"options": {"device": {}}})

    device: str = ""
    sample_rate: int = 48000
    channels: int = 2
    frame_ms: int = 20


class AudioIn(ThreadedComponent[tuple[()], MicOutputs]):
    tags = Tag(io={"source"}, functionality={"audio"})
    description = "Captures audio from any system **input device**. The node exposes the same `audio` output as `Mic`, but adds a dynamic device dropdown so you can listen to virtual cables or other capture devices."

    def __init__(self, config: AudioInConfig = AudioInConfig()) -> None:
        super().__init__()
        self.config = config
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels
        self._frame_samples = int(self._sample_rate * self.config.frame_ms / 1000)
        self._device = coerce_sounddevice_device(self.config.device)

    @classmethod
    def get_options(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {"config": {"device": list_audio_input_devices()}}

    def run(self, inputs: tuple[()], outputs: MicOutputs) -> None:
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
            device=self._device,
            latency="low",
        ) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(self._frame_samples)
                outputs.audio.send(
                    AudioFrame.new(
                        data=data.copy(),
                        sample_rate=self._sample_rate,
                        channels=self._channels,
                    )
                )
