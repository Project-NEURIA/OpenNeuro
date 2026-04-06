from __future__ import annotations

import subprocess
import sys
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

_USE_PAREC = sys.platform == "linux"


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
        if _USE_PAREC:
            from src.lib.audio.devices import list_pulse_sources

            return {"config": {"device": list_pulse_sources()}}
        return {"config": {"device": list_audio_input_devices()}}

    def _run_parec(self, outputs: MicOutputs) -> None:
        device = self.config.device or None
        cmd = [
            "parec",
            "--format=s16le",
            f"--rate={self._sample_rate}",
            f"--channels={self._channels}",
        ]
        if device:
            cmd.append(f"--device={device}")

        frame_bytes = self._frame_samples * self._channels * 2
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            while not self.stop_event.is_set():
                data = proc.stdout.read(frame_bytes)  # type: ignore[union-attr]
                if not data:
                    break
                outputs.audio.send(
                    AudioFrame.new(
                        data=data,
                        sample_rate=self._sample_rate,
                        channels=self._channels,
                    )
                )
        finally:
            proc.terminate()
            proc.wait()

    def _run_sounddevice(self, outputs: MicOutputs) -> None:
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

    def run(self, inputs: tuple[()], outputs: MicOutputs) -> None:
        if _USE_PAREC:
            self._run_parec(outputs)
        else:
            self._run_sounddevice(outputs)
