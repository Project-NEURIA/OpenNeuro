from __future__ import annotations

import threading
from typing import NamedTuple

from pydantic import BaseModel

from fishaudio import FishAudio  # type: ignore[import-untyped]

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent
from src.core.frames import AudioFrame, EOS, InterruptFrame, TextFrame


class FishTTSConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.fish.audio"
    reference_id: str = "258ed8fe8f2347f6b7d56bafc2041a3a"
    model: str = "s2-pro"


class FishTTSInputs(NamedTuple):
    text: Receiver[TextFrame | EOS]
    interrupt: Receiver[InterruptFrame] | None = None


class FishTTSOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class FishTTS(ThreadedComponent[FishTTSInputs, FishTTSOutputs]):
    """Text-to-Speech using Fish Audio WebSocket streaming."""

    description = "Streaming TTS via Fish Audio"

    def __init__(self, config: FishTTSConfig) -> None:
        super().__init__()
        self.config = config
        self._client = FishAudio(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def run(self, inputs: FishTTSInputs, outputs: FishTTSOutputs) -> None:
        interrupted = threading.Event()

        if inputs.interrupt is not None:
            interrupt_recv = inputs.interrupt

            def handle_interrupts() -> None:
                for frame in interrupt_recv(self):
                    if frame is None:
                        break
                    interrupted.set()

            threading.Thread(target=handle_interrupts, daemon=True).start()

        def text_stream():
            for frame in inputs.text(self):
                if frame is None or interrupted.is_set():
                    break
                yield frame.text

        remainder = b""
        for chunk in self._client.tts.stream_websocket(
            text_stream(),
            reference_id=self.config.reference_id,
            format="pcm",
            latency="balanced",
            model=self.config.model,
        ):
            if interrupted.is_set():
                break
            if chunk:
                data = remainder + chunk
                usable = len(data) - (len(data) % 2)
                if usable > 0:
                    outputs.audio.send(
                        AudioFrame.new(
                            data=data[:usable], sample_rate=44100, channels=1
                        )
                    )
                remainder = data[usable:]
