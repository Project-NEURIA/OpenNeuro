from __future__ import annotations

import re
import socket
import struct
import threading
import time
from collections import deque
from typing import Any, TypedDict

from pydantic import BaseModel

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import TextFrame, InterruptFrame

GENERATE_END_FLAG = "[END_OF_GENERATE]"


def _pad4(data: bytes) -> bytes:
    pad = (4 - (len(data) % 4)) % 4
    return data + (b"\x00" * pad)


def _osc_string(value: str) -> bytes:
    return _pad4(value.encode("utf-8") + b"\x00")


def _osc_arg(arg: Any) -> tuple[str, bytes]:
    if isinstance(arg, bool):
        return ("T" if arg else "F", b"")
    if isinstance(arg, int):
        return ("i", struct.pack(">i", arg))
    if isinstance(arg, float):
        return ("f", struct.pack(">f", arg))
    if isinstance(arg, str):
        return ("s", _osc_string(arg))
    raise TypeError(f"Unsupported OSC arg type: {type(arg)}")


def _osc_message(address: str, args: list[Any]) -> bytes:
    addr = address if address.startswith("/") else f"/{address}"
    tags: list[str] = []
    data: list[bytes] = []
    for arg in args:
        tag, payload = _osc_arg(arg)
        tags.append(tag)
        data.append(payload)
    type_tags = "," + "".join(tags)
    return _osc_string(addr) + _osc_string(type_tags) + b"".join(data)


class _OscClient:
    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()

    def send_message(self, address: str, args: list[Any]) -> None:
        msg = _osc_message(address, args)
        with self._lock:
            self._sock.sendto(msg, self._addr)

    def close(self) -> None:
        with self._lock:
            self._sock.close()


class OSCChatboxConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000
    chatbox_send: bool = True
    max_chars: int = 144
    text_flush_ms: int = 600
    chars_per_second: float = 12.0
    min_display_ms: int = 1200
    max_display_ms: int = 6000
    clear_on_last: bool = True


class OSCChatboxOutputs(TypedDict):
    pass


class OSCChatbox(
    Component[[Channel[TextFrame], Channel[InterruptFrame]], OSCChatboxOutputs]
):
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

    def get_output_channels(self) -> OSCChatboxOutputs:
        return {}

    def _send_chatbox(self, text: str, *, reset: bool = False) -> None:
        msg = text.strip()
        if not msg and not reset:
            return
        if reset:
            self._client.send_message("/chatbox/input", [msg, self.config.chatbox_send, True])
        else:
            self._client.send_message("/chatbox/input", [msg, self.config.chatbox_send])

    def _split_text(self, text: str) -> list[str]:
        raw = text.strip()
        if not raw:
            return []
        sentences = [p.strip() for p in re.split(r"(?<=[\.\!\?。！？…])\s+", raw) if p.strip()]
        max_chars = max(self.config.max_chars, 1)

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
        cps = max(self.config.chars_per_second, 1.0)
        est_ms = (len(text) / cps) * 1000.0
        est_ms = max(self.config.min_display_ms, est_ms)
        est_ms = min(self.config.max_display_ms, est_ms)
        return est_ms / 1000.0

    def _enqueue_text(self, text: str) -> None:
        chunks = self._split_text(text)
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

    def _text_loop(self, text: Channel[TextFrame]) -> None:
        for frame in text.stream(self):
            if frame is None:
                break
            token = frame.get()
            if token == GENERATE_END_FLAG:
                self._flush_text_buffer()
                continue
            with self._text_lock:
                self._text_buffer += token
                self._last_text_time = time.monotonic()

    def _text_flush_monitor(self) -> None:
        if self.config.text_flush_ms <= 0:
            return
        threshold = self.config.text_flush_ms / 1000.0
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
                if is_last and self.config.clear_on_last:
                    self._send_chatbox("", reset=True)

    def _interrupt_loop(self, interrupt: Channel[InterruptFrame]) -> None:
        for frame in interrupt.stream(self):
            if frame is None:
                break
            with self._text_lock:
                self._text_buffer = ""
            with self._send_lock:
                self._send_queue.clear()

    def run(
        self,
        text: Channel[TextFrame] | None = None,
        interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        threads: list[threading.Thread] = []

        if text is not None:
            threads.append(threading.Thread(target=self._text_loop, args=(text,)))
            threads.append(threading.Thread(target=self._text_flush_monitor))
            threads.append(threading.Thread(target=self._send_worker))
        if interrupt is not None:
            threads.append(threading.Thread(target=self._interrupt_loop, args=(interrupt,)))

        if not threads:
            while not self.stop_event.is_set():
                time.sleep(0.1)
            return

        for t in threads:
            t.daemon = True
            t.start()
        for t in threads:
            t.join()
