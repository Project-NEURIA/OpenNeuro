from __future__ import annotations

import base64
import json
import os
import re
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TypedDict, Callable

import requests

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import AudioFrame, InterruptFrame, TextFrame
from src.core.config import BaseConfig

GENERATE_END_FLAG = "[END_OF_GENERATE]"


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


class TTSConfig(BaseConfig):
    url: str = "https://api.inworld.ai/v1/tts/stream"
    voice_id: str = "default"
    model_id: str = "default"


class TTSOutputs(TypedDict):
    audio: Channel[AudioFrame]
    text: Channel[TextFrame]
    interrupt: Channel[InterruptFrame]


class TTS(Component[[Channel[TextFrame], Channel[InterruptFrame]], TTSOutputs]):
    """Text-to-Speech component using Inworld API."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        super().__init__(config or TTSConfig())
        self.config: TTSConfig
        self._output_audio = Channel[AudioFrame](name="audio")
        self._output_text = Channel[TextFrame](name="text")
        self._output_interrupt = Channel[InterruptFrame](name="interrupt")

        self._stream_filter = StreamFilter()
        self._generation = 0
        self._gen_lock = threading.Lock()
        self._task_queue: Queue[tuple[int, str]] = Queue()

    def get_output_channels(self) -> TTSOutputs:
        return {
            "audio": self._output_audio,
            "text": self._output_text,
            "interrupt": self._output_interrupt,
        }

    def _worker(self) -> None:
        print("[TTS] Worker thread started")
        while not self.stop_event.is_set():
            try:
                gen, text = self._task_queue.get(timeout=0.1)
                with self._gen_lock:
                    if gen != self._generation:
                        continue

                cred = os.getenv("INWORLD_API_CRED")
                if not cred:
                    print("[TTS] INWORLD_API_CRED not set")
                    continue

                headers = {
                    "Authorization": f"Basic {cred}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "text": text,
                    "voiceId": self.config.voice_id,
                    "modelId": self.config.model_id,
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
                            # Use AudioFrame with data getter logic
                            self._output_audio.send(
                                AudioFrame(
                                    display_name="tts_audio",
                                    data=raw[44:],
                                    sample_rate=48000,
                                    channels=1,
                                )
                            )

                    with self._gen_lock:
                        if gen == self._generation:
                            self._output_text.send(
                                TextFrame(display_name="tts_text", text=text)
                            )
                except Exception as e:
                    print(f"[TTS] Generation error: {e}")
            except Empty:
                continue

    def run(
        self,
        text_input: Channel[TextFrame] | None = None,
        interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        print("[TTS] Starting TTS")
        worker_thread = threading.Thread(target=self._worker, daemon=True)
        worker_thread.start()

        def handle_interrupts():
            if not interrupt:
                return
            for frame in interrupt.stream(self):
                if frame is None:
                    break

                print(f"[TTS] Interrupt received: {frame.get()}")

                with self._gen_lock:
                    self._generation += 1

                # Forward the interrupt
                self._output_interrupt.send(frame)

                self._stream_filter = StreamFilter()
                while not self._task_queue.empty():
                    try:
                        self._task_queue.get_nowait()
                    except Empty:
                        break

        threading.Thread(target=handle_interrupts, daemon=True).start()

        if text_input:
            for frame in text_input.stream(self):
                if frame is None:
                    break
                # Use .get() instead of .text
                t = frame.get()
                out = (
                    self._stream_filter.feed("", force=True)
                    if t == GENERATE_END_FLAG
                    else self._stream_filter.feed(t)
                )
                if out and out.strip():
                    with self._gen_lock:
                        gen = self._generation
                    self._task_queue.put((gen, out))

        worker_thread.join(timeout=1)
        print("[TTS] TTS stopped")
