from __future__ import annotations

import sounddevice as sd

from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import AudioFrame


class Mic(Component[()]):

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
        self._output = Channel[AudioFrame]()

    def set_input_channels(self) -> None:
        pass

    def get_output_channels(self) -> tuple[Channel[AudioFrame]]:
        return (self._output,)

    def run(self) -> None:
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
        ) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(self._frame_samples)
                
                # Create AudioFrame with audio data
                audio_frame = AudioFrame(
                    pcm16_data=data.tobytes(),
                    sample_rate=self._sample_rate,
                    channels=self._channels
                )
                
                self._output.send(audio_frame)
