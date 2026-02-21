from __future__ import annotations

import socket
import struct
import threading
import time
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
    chatbox_max_chars: int = 144
    text_flush_ms: int = 1200


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

    def stop(self) -> None:
        self._client.close()
        super().stop()

    def get_output_channels(self) -> OSCChatboxOutputs:
        return {}

    def _send_chatbox(self, text: str) -> None:
        msg = text.strip()
        if not msg:
            return
        if self.config.chatbox_max_chars > 0:
            msg = msg[: self.config.chatbox_max_chars]
        self._client.send_message("/chatbox/input", [msg, self.config.chatbox_send])

    def _flush_text_buffer(self) -> None:
        with self._text_lock:
            text = self._text_buffer
            self._text_buffer = ""
        if text:
            self._send_chatbox(text)

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
                self._send_chatbox(text)

    def _interrupt_loop(self, interrupt: Channel[InterruptFrame]) -> None:
        for frame in interrupt.stream(self):
            if frame is None:
                break
            with self._text_lock:
                self._text_buffer = ""

    def run(
        self,
        text: Channel[TextFrame] | None = None,
        interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        threads: list[threading.Thread] = []

        if text is not None:
            threads.append(threading.Thread(target=self._text_loop, args=(text,)))
            threads.append(threading.Thread(target=self._text_flush_monitor))
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
