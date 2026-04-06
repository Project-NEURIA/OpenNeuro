from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import sounddevice as sd
from pydantic import BaseModel, ConfigDict

from src.lib.audio.devices import (
    coerce_sounddevice_device,
    list_audio_output_devices,
)
from src.lib.audio.speaker import SpeakerInputs
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioDataFormat

_USE_PAPLAY = sys.platform == "linux"


class AudioOutConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"options": {"device": {}}})

    device: str = ""
    sample_rate: int = 48000
    channels: int = 1


class AudioOut(ThreadedComponent[SpeakerInputs, tuple[()]]):
    tags = Tag(io={"sink"}, functionality={"audio"})
    description = "Plays `AudioFrame` data to any system **output device**. The node exposes the same `audio` input as `Speaker`, but adds a dynamic device dropdown so you can target virtual cables or specific playback devices."

    def __init__(self, config: AudioOutConfig = AudioOutConfig()) -> None:
        super().__init__()
        self.config = config
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels
        self._device = coerce_sounddevice_device(self.config.device)

    @classmethod
    def get_options(cls, values: dict[str, Any]) -> dict[str, Any]:
        if _USE_PAPLAY:
            from src.lib.audio.devices import list_pulse_sinks

            return {"config": {"device": list_pulse_sinks()}}
        return {"config": {"device": list_audio_output_devices()}}

    def _run_paplay(self, inputs: SpeakerInputs) -> None:
        device = self.config.device or None
        cmd = [
            "paplay",
            "--raw",
            "--format=s16le",
            f"--rate={self._sample_rate}",
            f"--channels={self._channels}",
        ]
        if device:
            cmd.append(f"--device={device}")

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for frame in inputs.audio:
                if frame is None:
                    break
                pcm_bytes = frame.get(
                    sample_rate=self._sample_rate,
                    num_channels=self._channels,
                    data_format=AudioDataFormat.PCM16,
                )
                try:
                    proc.stdin.write(pcm_bytes)  # type: ignore[union-attr]
                except BrokenPipeError:
                    break
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait()

    def _run_sounddevice(self, inputs: SpeakerInputs) -> None:
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            device=self._device,
            latency="low",
        ) as stream:
            for frame in inputs.audio:
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

    def run(self, inputs: SpeakerInputs, outputs: tuple[()]) -> None:
        if _USE_PAPLAY:
            self._run_paplay(inputs)
        else:
            self._run_sounddevice(inputs)
