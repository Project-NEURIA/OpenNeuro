from __future__ import annotations

import base64
import json
import os
import re
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, NamedTuple

import requests
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import AudioFrame, EOS, InterruptFrame, TextFrame


def cut_space(buf: str) -> int:
    return max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))


_SENT_END = re.compile(r"""[.!?…]+["'\)\]\}]*($|\s)""")


def cut_sentence(buf: str) -> int:
    last = -1
    for m in _SENT_END.finditer(buf):
        cut = m.end() - 1
        while cut >= 0 and buf[cut].isspace():
            cut -= 1
        last = max(last, cut)
    return last


@dataclass
class StreamFilter:
    speak_buf: str = ""
    in_square: int = 0
    in_paren: int = 0
    in_angle: int = 0
    in_bold: bool = False
    in_italic: bool = False
    cut_fn: Callable[[str], int] = field(default=cut_sentence)

    def feed(self, token: str, force: bool = False) -> str:
        self._consume(token)
        if force:
            out, self.speak_buf = self.speak_buf, ""
            return out

        cut = self.cut_fn(self.speak_buf)
        if cut < 0:
            return ""
        out = self.speak_buf[: cut + 1]
        self.speak_buf = self.speak_buf[cut + 1 :]
        return out

    def _consume(self, token: str) -> None:
        i = 0
        while i < len(token):
            ch = token[i]
            if ch == "*" and i + 1 < len(token) and token[i + 1] == "*":
                self.in_bold = not self.in_bold
                i += 2
                continue
            if ch == "*":
                self.in_italic = not self.in_italic
                i += 1
                continue
            if not (self.in_bold or self.in_italic):
                if ch == "[":
                    self.in_square += 1
                    i += 1
                    continue
                if ch == "]" and self.in_square:
                    self.in_square -= 1
                    i += 1
                    continue
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
                self.in_square
                or self.in_paren
                or self.in_angle
                or self.in_bold
                or self.in_italic
            ):
                i += 1
                continue
            self.speak_buf += ch
            i += 1


class TTSConfig(BaseModel):
    api_key_env_var: str = "INWORLD_API_CRED"
    url: str = "https://api.inworld.ai/tts/v1/voice:stream"
    voice_id: str = "Ashley"
    model_id: str = "inworld-tts-1.5-max"


class TTSInputs(NamedTuple):
    text: Receiver[TextFrame | EOS]
    interrupt: Receiver[InterruptFrame] | None = None


class TTSOutputs(NamedTuple):
    audio: Sender[AudioFrame]
    text: Sender[TextFrame]


class TTS(Component[TTSInputs, TTSOutputs]):
    """Text-to-Speech component using Inworld API."""

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

                cred = os.getenv("INWORLD_API_KEY") or os.getenv(self.config.api_key_env_var)
                if not cred:
                    raise ValueError(f"Environment variable {self.config.api_key_env_var} must be set")

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
                for frame in interrupt_recv(self):
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

        for frame in inputs.text(self):
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
