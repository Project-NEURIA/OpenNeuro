from __future__ import annotations

import base64
import json
import os
import threading
import time
from queue import Empty, Queue
from typing import NamedTuple, Callable

import httpx
import ormsgpack
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent
from src.core.utils import StreamFilter, cut_sentence
from src.core.frames import AudioFrame, EOS, InterruptFrame, TextFrame


class FishStreamFilter(StreamFilter):
    """Modified filter for Fish TTS that preserves [tags]."""

    def _consume(self, token: str) -> None:
        i = 0
        while i < len(token):
            ch = token[i]
            # Toggle bold (**)
            if ch == "*" and i + 1 < len(token) and token[i + 1] == "*":
                self.in_bold = not self.in_bold
                i += 2
                continue
            # Toggle italic (*)
            if ch == "*":
                self.in_italic = not self.in_italic
                i += 1
                continue

            # We DO NOT filter square brackets for Fish S2-Pro tags
            if not (self.in_bold or self.in_italic):
                if ch == "(":
                    self.in_paren += 1
                    i += 1
                    continue
                if ch == ")" and self.in_paren:
                    self.in_paren -= 1
                    i += 1
                    continue
                if ch == "<":
                    self.in_angle += 1
                    i += 1
                    continue
                if ch == ">" and self.in_angle:
                    self.in_angle -= 1
                    i += 1
                    continue

            if (
                self.in_paren
                or self.in_angle
                or self.in_bold
                or self.in_italic
            ):
                i += 1
                continue

            self.speak_buf += ch
            i += 1


class TTSConfig(BaseModel):
    api_key_env_var: str = "FISH_API_KEY"
    url: str = "https://api.fish.audio/v1/tts"
    reference_id: str = "258ed8fe8f2347f6b7d56bafc2041a3a"
    pitch: float = 0.0


class TTSInputs(NamedTuple):
    text: Receiver[TextFrame | EOS]
    interrupt: Receiver[InterruptFrame] | None = None


class TTSOutputs(NamedTuple):
    audio: Sender[AudioFrame]
    text: Sender[TextFrame]


class TTS(PrimitiveComponent[TTSInputs, TTSOutputs]):
    """Text-to-Speech component using Fish Audio S2-Pro API."""

    def __init__(self, config: TTSConfig) -> None:
        super().__init__()
        self.config: TTSConfig = config
        self._stream_filter = FishStreamFilter()
        self._generation = 0
        self._gen_lock = threading.Lock()
        self._task_queue: Queue[tuple[int, str]] = Queue()
        # Persistent client for connection pooling and streaming
        self._client = httpx.Client(timeout=30.0)

    def _worker(self, outputs: TTSOutputs) -> None:
        print("[FishTTS] Worker thread started")
        while not self.stop_event.is_set():
            try:
                gen, text = self._task_queue.get(timeout=0.1)
                with self._gen_lock:
                    if gen != self._generation:
                        continue

                if text == "[END_OF_TURN]":
                    outputs.text.send(TextFrame.new(text=text))
                    continue

                api_key = os.getenv(self.config.api_key_env_var)
                if not api_key:
                    print(f"[FishTTS] Error: {self.config.api_key_env_var} not set")
                    continue

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/msgpack",
                }

                # Fish returns raw PCM if specified
                payload = {
                    "text": text,
                    "reference_id": self.config.reference_id,
                    "format": "pcm",
                    "streaming": True,
                    "use_memory_cache": "on",
                    "normalize": True,
                    "latency": "balanced",
                }

                try:
                    # Optimized streaming with httpx
                    with self._client.stream(
                        "POST",
                        self.config.url,
                        content=ormsgpack.packb(payload),
                        headers=headers,
                        params={"format": "msgpack"}
                    ) as r:
                        r.raise_for_status()

                        sr = int(44100 * (2 ** (self.config.pitch / 12.0))) if self.config.pitch != 0 else 44100

                        for chunk in r.iter_bytes():
                            with self._gen_lock:
                                if gen != self._generation:
                                    break
                            
                            if chunk:
                                outputs.audio.send(
                                    AudioFrame.new(
                                        data=chunk,
                                        sample_rate=sr,
                                        channels=1,
                                    )
                                )

                        with self._gen_lock:
                            if gen == self._generation:
                                outputs.text.send(TextFrame.new(text=text))

                except Exception as e:
                    print(f"[FishTTS] Generation error: {e}")
            except Empty:
                continue

    def run(self, inputs: TTSInputs, outputs: TTSOutputs) -> None:
        print("[FishTTS] Starting TTS")
        worker_thread = threading.Thread(
            target=self._worker, args=(outputs,), daemon=True
        )
        worker_thread.start()

        if inputs.interrupt is not None:
            interrupt_recv = inputs.interrupt

            def handle_interrupts() -> None:
                for frame in interrupt_recv(self):
                    if frame is None:
                        break

                    print(f"[FishTTS] Interrupt received")

                    with self._gen_lock:
                        self._generation += 1

                    self._stream_filter = FishStreamFilter()
                    while not self._task_queue.empty():
                        try:
                            self._task_queue.get_nowait()
                        except Empty:
                            break

            threading.Thread(target=handle_interrupts, daemon=True).start()

        for frame in inputs.text(self):
            if frame is None:
                break
            if frame is EOS.END:
                out = self._stream_filter.feed("", force=True)
                if out and out.strip():
                    with self._gen_lock:
                        gen = self._generation
                    self._task_queue.put((gen, out))
                
                with self._gen_lock:
                    gen = self._generation
                self._task_queue.put((gen, "[END_OF_TURN]"))
                self._stream_filter = FishStreamFilter()
            else:
                out = self._stream_filter.feed(frame.text)
                if out and out.strip():
                    with self._gen_lock:
                        gen = self._generation
                    self._task_queue.put((gen, out))

        worker_thread.join(timeout=1)
        print("[FishTTS] TTS stopped")
