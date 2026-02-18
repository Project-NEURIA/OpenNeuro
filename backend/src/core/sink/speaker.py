from __future__ import annotations

from typing import TypedDict

import numpy as np
import sounddevice as sd

from src.core.component import Component
from src.core.channel import Channel


from src.core.config import BaseConfig


from src.core.frames import AudioFrame, AudioDataFormat


class SpeakerConfig(BaseConfig):
    sample_rate: int = 24000
    channels: int = 1


class SpeakerOutputs(TypedDict):
    pass


class Speaker(Component[[Channel[AudioFrame]], SpeakerOutputs]):
    def __init__(self, config: SpeakerConfig | None = None) -> None:
        super().__init__(config or SpeakerConfig())
        self.config: SpeakerConfig  # Type hint for IDE
        self._sample_rate = self.config.sample_rate
        self._channels = self.config.channels

    def get_output_channels(self) -> SpeakerOutputs:
        return {}

    def run(self, audio: Channel[AudioFrame] | None = None) -> None:
        if not audio:
            print("[Speaker] No audio channel connected, skipping")
            return

        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for frame in audio.stream(self):
                if frame is None:
                    break

                # AudioFrame.get handles resampling and channel mixing automatically
                pcm_bytes = frame.get(
                    sample_rate=self._sample_rate,
                    num_channels=self._channels,
                    data_format=AudioDataFormat.PCM16,
                )

                stream.write(
                    np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, self._channels)
                )
