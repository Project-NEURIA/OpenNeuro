from __future__ import annotations

import os
import threading
from typing import NamedTuple

from pydantic import BaseModel

from fishaudio import FishAudio  # type: ignore[import-untyped]

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioFrame, EOS, InterruptFrame, TextFrame


class FishTTSConfig(BaseModel):
    api_key_env_var: str = "FISH_API_KEY"
    base_url: str = "https://api.fish.audio"
    reference_id: str = "258ed8fe8f2347f6b7d56bafc2041a3a"
    model: str = "s2-pro"


class FishTTSInputs(NamedTuple):
    text: Receiver[TextFrame | EOS]
    interrupt: Receiver[InterruptFrame] | None = None


class FishTTSOutputs(NamedTuple):
    audio: Sender[AudioFrame]
    done_speaking: Sender[TextFrame] | None = None


class FishTTS(ThreadedComponent[FishTTSInputs, FishTTSOutputs]):
    """Text-to-Speech using Fish Audio WebSocket streaming."""

    tags = Tag(io={"conduit"}, functionality={"audio"})
    description = "**Streaming TTS** via *Fish Audio* WebSocket API. Converts `TextFrame` tokens into `AudioFrame` output using configurable *voice references* and the `s2-pro` model."

    def __init__(self, config: FishTTSConfig) -> None:
        super().__init__()
        self.config = config
        api_key = os.getenv(config.api_key_env_var, "")
        self._client = FishAudio(
            api_key=api_key,
            base_url=config.base_url,
        )

    def run(self, inputs: FishTTSInputs, outputs: FishTTSOutputs) -> None:
        interrupted = threading.Event()

        if inputs.interrupt is not None:
            interrupt_recv = inputs.interrupt

            def handle_interrupts() -> None:
                for frame in interrupt_recv:
                    if frame is None:
                        break
                    interrupted.set()

            threading.Thread(target=handle_interrupts, daemon=True).start()

        # Outer loop: one websocket call per utterance
        for frame in inputs.text:
            if frame is None or self.stop_event.is_set():
                break

            # Collect tokens until EOS
            tokens: list[str] = [frame.text]
            inputs.text.blocking = False
            for extra in inputs.text:
                if extra is None:
                    break
                if isinstance(extra, EOS):
                    break
                tokens.append(extra.text)
            inputs.text.blocking = True

            if interrupted.is_set():
                interrupted.clear()
                continue

            def token_iter():
                for t in tokens:
                    if interrupted.is_set() or self.stop_event.is_set():
                        break
                    yield t

            import time as _time

            remainder = b""
            first_chunk_time: float | None = None
            total_samples = 0
            sample_rate = 44100

            for chunk in self._client.tts.stream_websocket(
                token_iter(),
                reference_id=self.config.reference_id,
                format="pcm",
                latency="balanced",
                model=self.config.model,
            ):
                if interrupted.is_set() or self.stop_event.is_set():
                    break
                if chunk:
                    data = remainder + chunk
                    usable = len(data) - (len(data) % 2)
                    if usable > 0:
                        if first_chunk_time is None:
                            first_chunk_time = _time.monotonic()
                        total_samples += usable // 2  # 16-bit PCM = 2 bytes per sample
                        outputs.audio.send(
                            AudioFrame.new(
                                data=data[:usable], sample_rate=sample_rate, channels=1
                            )
                        )
                    remainder = data[usable:]

            # Wait until audio finishes playing, then signal done
            if outputs.done_speaking is not None and first_chunk_time is not None:
                audio_duration = total_samples / sample_rate
                elapsed = _time.monotonic() - first_chunk_time
                remaining = audio_duration - elapsed
                if remaining > 0:
                    self.stop_event.wait(remaining)
                if not self.stop_event.is_set():
                    outputs.done_speaking.send(TextFrame.new(text="done"))
