from __future__ import annotations

import base64
import json
import os
import threading
from queue import Empty, Queue
from typing import NamedTuple

import requests
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.utils import StreamFilter
from src.core.frames import AudioFrame, EOS, InterruptFrame, TextFrame


class TTSConfig(BaseModel):
    api_key_env_var: str = "INWORLD_API_KEY"
    url: str = "https://api.inworld.ai/tts/v1/voice:stream"
    voice_id: str = "Ashley"
    model_id: str = "inworld-tts-1.5-max"


class TTSInputs(NamedTuple):
    text: Receiver[TextFrame | EOS]
    interrupt: Receiver[InterruptFrame] | None = None


class TTSOutputs(NamedTuple):
    audio: Sender[AudioFrame]
    text: Sender[TextFrame]


class TTS(ThreadedComponent[TTSInputs, TTSOutputs]):
    description = "Converts text to **speech audio** using the *Inworld TTS* API. Streams `TextFrame` tokens into synthesized `AudioFrame` output with configurable *voice* and *model* selection."

    """Text-to-Speech component using Inworld API."""

    tags = Tag(io={"conduit"}, functionality={"audio"}, gpu={"cpu", "nvidia", "apple"})

    def __init__(self, config: TTSConfig) -> None:
        super().__init__()
        self.config: TTSConfig = config
        self._stream_filter = StreamFilter()
        self._generation = 0
        self._gen_lock = threading.Lock()
        self._task_queue: Queue[tuple[int, str]] = Queue()

    def _worker(self, outputs: TTSOutputs) -> None:
        print("[TTS] Worker thread started")
        while not self.stop_event.is_set():
            try:
                gen, text = self._task_queue.get(timeout=0.1)
                with self._gen_lock:
                    if gen != self._generation:
                        continue

                cred = os.getenv(self.config.api_key_env_var)
                print(
                    f"[TTS] Using {self.config.api_key_env_var}: {cred[:8]}..."
                    if cred
                    else f"[TTS] {self.config.api_key_env_var} not set"
                )
                if not cred:
                    raise ValueError(
                        f"Environment variable {self.config.api_key_env_var} must be set"
                    )

                headers = {
                    "Authorization": f"Basic {cred}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "text": text,
                    "voice_id": self.config.voice_id,
                    "model_id": self.config.model_id,
                    "audio_config": {
                        "audio_encoding": "LINEAR16",
                        "sample_rate_hertz": 48000,
                    },
                }

                try:
                    r = requests.post(
                        self.config.url,
                        json=payload,
                        headers=headers,
                        stream=True,
                        timeout=10,
                    )
                    r.raise_for_status()
                    for line in r.iter_lines():
                        with self._gen_lock:
                            if gen != self._generation:
                                break
                        if not line:
                            continue
                        msg = json.loads(line)
                        raw = base64.b64decode(msg["result"]["audioContent"])
                        if len(raw) > 44:
                            outputs.audio.send(
                                AudioFrame.new(
                                    data=raw[44:],
                                    sample_rate=48000,
                                    channels=1,
                                )
                            )

                    with self._gen_lock:
                        if gen == self._generation:
                            outputs.text.send(TextFrame.new(text=text))
                except Exception as e:
                    print(f"[TTS] Generation error: {e}")
            except Empty:
                continue

    def run(self, inputs: TTSInputs, outputs: TTSOutputs) -> None:
        print("[TTS] Starting TTS")
        worker_thread = threading.Thread(
            target=self._worker, args=(outputs,), daemon=True
        )
        worker_thread.start()

        if inputs.interrupt is not None:
            interrupt_recv = inputs.interrupt

            def handle_interrupts() -> None:
                for frame in interrupt_recv:
                    if frame is None:
                        break

                    print(f"[TTS] Interrupt received: {frame.reason}")

                    with self._gen_lock:
                        self._generation += 1

                    self._stream_filter = StreamFilter()
                    while not self._task_queue.empty():
                        try:
                            self._task_queue.get_nowait()
                        except Empty:
                            break

            threading.Thread(target=handle_interrupts, daemon=True).start()

        for frame in inputs.text:
            if frame is None:
                break
            if frame is EOS.END:
                out = self._stream_filter.feed("", force=True)
            else:
                out = self._stream_filter.feed(frame.text)
            if out and out.strip():
                with self._gen_lock:
                    gen = self._generation
                self._task_queue.put((gen, out))

        worker_thread.join(timeout=1)
        print("[TTS] TTS stopped")
