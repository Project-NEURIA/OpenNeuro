from __future__ import annotations


import numpy as np
import sounddevice as sd

from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import AudioFrame


class Speaker(Component[Channel[AudioFrame]]):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels
        self._input_channel: Channel[AudioFrame] | None = None

    def get_output_channels(self) -> tuple[()]:
        return ()

    def set_input_channels(self, t1: Channel[AudioFrame]) -> None:
        self._input_channel = t1

    def run(self) -> None:
        if self._input_channel is None:
            # Wait until input channel is set or stop event
            while self._input_channel is None and not self.stop_event.is_set():
                self.stop_event.wait(0.1)
            
            if self.stop_event.is_set() or self._input_channel is None:
                return
        
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
        ) as stream:
            for audio_frame in self._input_channel.stream(self.stop_event):
                if audio_frame is None:
                    break
                
                # Skip non-audio frames (e.g., TextFrame from TTS completion)
                if not isinstance(audio_frame, AudioFrame):
                    continue
                
                # Resample if needed
                if audio_frame.sample_rate != self._sample_rate:
                    audio_frame = audio_frame.resample(self._sample_rate)
                
                # Convert to numpy array for playback
                data = np.frombuffer(audio_frame.pcm16_data, dtype=np.int16)
                
                # Handle channel conversion
                if audio_frame.channels == 1 and self._channels == 2:
                    # Convert mono to stereo
                    data = np.column_stack([data, data])
                elif audio_frame.channels == 2 and self._channels == 1:
                    # Convert stereo to mono by averaging
                    data = data.reshape(-1, 2).mean(axis=1, dtype=np.int16)
                elif audio_frame.channels == 2 and self._channels == 2:
                    # Keep stereo
                    data = data.reshape(-1, 2)
                else:
                    # Keep mono
                    data = data.reshape(-1, 1)
                
                stream.write(data)
