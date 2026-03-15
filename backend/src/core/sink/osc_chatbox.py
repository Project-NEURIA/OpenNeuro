from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any, NamedTuple

from pydantic import BaseModel
from pythonosc.udp_client import SimpleUDPClient

from src.core.component import PrimitiveComponent, Tag
from src.core.channel import Receiver
from src.core.frames import EOS, TextFrame, InterruptFrame


class _OscClient:
    """
    Thin wrapper around python-osc with a send lock.
    """

    def __init__(self, host: str, port: int) -> None:
        self._client = SimpleUDPClient(host, port)
        self._lock = threading.Lock()

    def send_message(self, address: str, args: list[Any]) -> None:
        with self._lock:
            self._client.send_message(address, args)

    def close(self) -> None:
        # SimpleUDPClient has no close() requirement; keep for interface symmetry.
        return


class OSCChatboxConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000

    # Keep the existing field name to avoid touching other code.
    # This value is used as the OSC "immediate" flag for /chatbox/input.
    chatbox_send: bool = True

    # New: control the "play_sound" flag for /chatbox/input.
    # Default False to avoid chatbox notification spam.
    play_sound: bool = False

    max_chars: int = 144
    text_flush_ms: int = 600
    chars_per_second: float = 12.0
    min_display_ms: int = 1200
    max_display_ms: int = 6000
    clear_on_last: bool = True


class OSCChatboxInputs(NamedTuple):
    text: Receiver[TextFrame | EOS] | None = None
    interrupt: Receiver[InterruptFrame] | None = None


class OSCChatbox(PrimitiveComponent[OSCChatboxInputs, tuple[()]]):
    _tags = Tag(io={"sink"}, functionality={"misc"})

    def __init__(self, config: OSCChatboxConfig) -> None:
        super().__init__()
        self.config = config
        self._client = _OscClient(self.config.host, self.config.port)

        self._text_lock = threading.Lock()
        self._text_buffer = ""
        self._last_text_time = 0.0

        self._send_lock = threading.Lock()
        self._send_queue: deque[tuple[str, float]] = deque()
        self._send_event = threading.Event()

    def stop(self) -> None:
        self._client.close()
        super().stop()

    def _send_chatbox(self, text: str, *, reset: bool = False) -> None:
        msg = text.strip()

        # If not reset, skip sending empty strings.
        if not msg and not reset:
            return

        # VRChat chatbox format:
        # /chatbox/input [text, immediate, play_sound]
        immediate = bool(self.config.chatbox_send)
        play_sound = bool(self.config.play_sound)

        # If reset, we still send, but msg will be "" which clears the chatbox.
        self._client.send_message("/chatbox/input", [msg, immediate, play_sound])

    def _split_text(self, text: str) -> list[str]:
        raw = text.strip()
        if not raw:
            return []

        sentences = [
            p.strip() for p in re.split(r"(?<=[\.\!\?。！？…])\s+", raw) if p.strip()
        ]

        max_chars = max(int(self.config.max_chars), 1)
        chunks: list[str] = []

        for sentence in sentences:
            if len(sentence) <= max_chars:
                chunks.append(sentence)
                continue

            words = sentence.split()
            current = ""

            for word in words:
                if not current:
                    current = word
                    continue

                if len(current) + 1 + len(word) <= max_chars:
                    current = f"{current} {word}"
                    continue

                chunks.append(current)

                if len(word) <= max_chars:
                    current = word
                else:
                    # Hard-split very long word
                    for i in range(0, len(word), max_chars):
                        chunks.append(word[i : i + max_chars])
                    current = ""

            if current:
                chunks.append(current)

        return chunks

    def _display_delay(self, text: str) -> float:
        cps = max(float(self.config.chars_per_second), 1.0)
        est_ms = (len(text) / cps) * 1000.0
        est_ms = max(int(self.config.min_display_ms), est_ms)
        est_ms = min(int(self.config.max_display_ms), est_ms)
        return float(est_ms) / 1000.0

    def _enqueue_text(self, text: str) -> None:
        chunks = self._split_text(text)
        if not chunks:
            return

        with self._send_lock:
            for chunk in chunks:
                self._send_queue.append((chunk, self._display_delay(chunk)))
            self._send_event.set()

    def _flush_text_buffer(self) -> None:
        with self._text_lock:
            text = self._text_buffer
            self._text_buffer = ""
        if text:
            self._enqueue_text(text)

    def _text_loop(self, text: Receiver[TextFrame]) -> None:
        for frame in text(self):
            if frame is None:
                break

            if frame is EOS.END:
                self._flush_text_buffer()
                continue

            with self._text_lock:
                self._text_buffer += frame.get()
                self._last_text_time = time.monotonic()

    def _text_flush_monitor(self) -> None:
        if int(self.config.text_flush_ms) <= 0:
            return

        threshold = float(self.config.text_flush_ms) / 1000.0

        while not self.stop_event.is_set():
            time.sleep(0.1)

            with self._text_lock:
                if not self._text_buffer:
                    continue

                idle = time.monotonic() - self._last_text_time
                if idle < threshold:
                    continue

                text = self._text_buffer
                self._text_buffer = ""

            if text:
                self._enqueue_text(text)

    def _send_worker(self) -> None:
        while not self.stop_event.is_set():
            self._send_event.wait(timeout=0.1)

            while True:
                with self._send_lock:
                    if not self._send_queue:
                        self._send_event.clear()
                        break

                    msg, delay = self._send_queue.popleft()
                    is_last = not self._send_queue

                self._send_chatbox(msg)

                if delay > 0:
                    time.sleep(delay)

                if is_last and bool(self.config.clear_on_last):
                    self._send_chatbox("", reset=True)

    def _interrupt_loop(self, interrupt: Receiver[InterruptFrame]) -> None:
        for frame in interrupt(self):
            if frame is None:
                break

            with self._text_lock:
                self._text_buffer = ""

            with self._send_lock:
                self._send_queue.clear()
                self._send_event.clear()

    def run(self, inputs: OSCChatboxInputs, outputs: tuple[()]) -> None:
        threads: list[threading.Thread] = []

        if inputs.text is not None:
            threads.append(
                threading.Thread(target=self._text_loop, args=(inputs.text,))
            )
            threads.append(threading.Thread(target=self._text_flush_monitor))
            threads.append(threading.Thread(target=self._send_worker))

        if inputs.interrupt is not None:
            threads.append(
                threading.Thread(target=self._interrupt_loop, args=(inputs.interrupt,))
            )

        if not threads:
            while not self.stop_event.is_set():
                time.sleep(0.1)
            return

        for t in threads:
            t.daemon = True
            t.start()

        for t in threads:
            t.join()
