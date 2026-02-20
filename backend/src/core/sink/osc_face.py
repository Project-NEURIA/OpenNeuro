from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any, TypedDict

import numpy as np
from pydantic import BaseModel

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import AudioFrame, AudioDataFormat, TextFrame, InterruptFrame

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


# Unified Expressions v2 presets (intended for /avatar/parameters/FT/v2/...)
# Notes:
# - EyeLid: 0.0..0.75 is openness, 0.75..1.0 is widen
# - "Small eyes": higher EyeSquint + moderately lower EyeLid
# - "Scared": high EyeLid (widen) + high BrowExpression + high PupilDilation + JawOpen


def expression_neutral() -> dict[str, float]:
    return {
        "v2/SmileFrownLeft": 0.0,
        "v2/SmileFrownRight": 0.0,
        "v2/MouthX": 0.0,
        "v2/BrowExpressionLeft": 0.0,
        "v2/BrowExpressionRight": 0.0,
        "v2/EyeLeftX": 0.0,
        "v2/EyeLeftY": 0.0,
        "v2/EyeRightX": 0.0,
        "v2/EyeRightY": 0.0,
        "v2/EyeLidLeft": 0.70,
        "v2/EyeLidRight": 0.70,
        "v2/EyeSquintLeft": 0.0,
        "v2/EyeSquintRight": 0.0,
        "v2/PupilDilation": 0.55,
        "v2/JawOpen": 0.0,
        "v2/MouthClosed": 0.0,
        "v2/LipFunnel": 0.0,
        "v2/LipPucker": 0.0,
    }


def expression_happy() -> dict[str, float]:
    return {
        "v2/SmileFrownLeft": 0.92,
        "v2/SmileFrownRight": 0.92,
        "v2/EyeSquintLeft": 0.28,
        "v2/EyeSquintRight": 0.28,
        "v2/EyeLidLeft": 0.65,
        "v2/EyeLidRight": 0.65,
        "v2/BrowExpressionLeft": 0.18,
        "v2/BrowExpressionRight": 0.18,
        "v2/JawOpen": 0.06,
        "v2/PupilDilation": 0.72,
    }


def expression_sad() -> dict[str, float]:
    return {
        "v2/SmileFrownLeft": -0.82,
        "v2/SmileFrownRight": -0.82,
        "v2/BrowExpressionLeft": 0.22,
        "v2/BrowExpressionRight": 0.22,
        "v2/EyeLeftY": -0.20,
        "v2/EyeRightY": -0.20,
        "v2/EyeLidLeft": 0.55,
        "v2/EyeLidRight": 0.55,
        "v2/EyeSquintLeft": 0.10,
        "v2/EyeSquintRight": 0.10,
        "v2/MouthClosed": 0.35,
        "v2/JawOpen": 0.04,
        "v2/PupilDilation": 0.82,
    }


def expression_angry() -> dict[str, float]:
    return {
        "v2/SmileFrownLeft": -0.38,
        "v2/SmileFrownRight": -0.38,
        "v2/BrowExpressionLeft": -0.95,
        "v2/BrowExpressionRight": -0.95,
        "v2/EyeSquintLeft": 0.75,
        "v2/EyeSquintRight": 0.75,
        "v2/EyeLidLeft": 0.60,
        "v2/EyeLidRight": 0.60,
        "v2/MouthClosed": 0.75,
        "v2/JawOpen": 0.02,
        "v2/PupilDilation": 0.25,
    }


def expression_surprised() -> dict[str, float]:
    return {
        "v2/BrowExpressionLeft": 0.75,
        "v2/BrowExpressionRight": 0.75,
        "v2/JawOpen": 0.55,
        "v2/MouthClosed": 0.0,
        "v2/LipFunnel": 0.28,
        "v2/LipPucker": 0.18,
        "v2/EyeLidLeft": 0.82,
        "v2/EyeLidRight": 0.82,
        "v2/PupilDilation": 0.92,
    }


def expression_scared() -> dict[str, float]:
    return {
        "v2/BrowExpressionLeft": 0.95,
        "v2/BrowExpressionRight": 0.95,
        "v2/EyeLidLeft": 0.95,
        "v2/EyeLidRight": 0.95,
        "v2/EyeSquintLeft": 0.0,
        "v2/EyeSquintRight": 0.0,
        "v2/PupilDilation": 1.0,
        "v2/JawOpen": 0.70,
        "v2/MouthClosed": 0.0,
        "v2/LipFunnel": 0.35,
        "v2/LipPucker": 0.10,
    }


def expression_thinking() -> dict[str, float]:
    return {
        "v2/SmileFrownLeft": 0.10,
        "v2/SmileFrownRight": 0.06,
        "v2/BrowExpressionLeft": 0.10,
        "v2/BrowExpressionRight": 0.20,
        "v2/EyeLeftX": 0.25,
        "v2/EyeRightX": 0.25,
        "v2/EyeSquintLeft": 0.15,
        "v2/EyeSquintRight": 0.10,
        "v2/EyeLidLeft": 0.62,
        "v2/EyeLidRight": 0.62,
        "v2/MouthClosed": 0.25,
        "v2/JawOpen": 0.02,
        "v2/PupilDilation": 0.55,
    }


def expression_shy() -> dict[str, float]:
    return {
        "v2/EyeLeftX": -0.55,
        "v2/EyeRightX": -0.55,
        "v2/EyeLeftY": -0.45,
        "v2/EyeRightY": -0.45,
        "v2/EyeSquintLeft": 0.58,
        "v2/EyeSquintRight": 0.58,
        "v2/EyeLidLeft": 0.55,
        "v2/EyeLidRight": 0.55,
        "v2/PupilDilation": 0.92,
        "v2/MouthClosed": 0.60,
        "v2/JawOpen": 0.01,
        "v2/SmileFrownLeft": 0.35,
        "v2/SmileFrownRight": 0.35,
        "v2/BrowExpressionLeft": 0.15,
        "v2/BrowExpressionRight": 0.15,
    }


def expression_smirk() -> dict[str, float]:
    return {
        "v2/MouthX": 0.45,
        "v2/SmileFrownLeft": 0.55,
        "v2/SmileFrownRight": 0.25,
        "v2/BrowExpressionLeft": -0.20,
        "v2/BrowExpressionRight": -0.30,
        "v2/EyeSquintLeft": 0.18,
        "v2/EyeSquintRight": 0.22,
        "v2/EyeLidLeft": 0.66,
        "v2/EyeLidRight": 0.66,
        "v2/PupilDilation": 0.40,
        "v2/MouthClosed": 0.10,
        "v2/JawOpen": 0.02,
    }


def expression_sleepy() -> dict[str, float]:
    return {
        "v2/EyeLeftY": -0.20,
        "v2/EyeRightY": -0.20,
        "v2/EyeLidLeft": 0.40,
        "v2/EyeLidRight": 0.40,
        "v2/EyeSquintLeft": 0.82,
        "v2/EyeSquintRight": 0.82,
        "v2/PupilDilation": 0.35,
        "v2/BrowExpressionLeft": -0.15,
        "v2/BrowExpressionRight": -0.15,
        "v2/MouthClosed": 0.15,
        "v2/JawOpen": 0.08,
        "v2/LipFunnel": 0.10,
    }


def expression_cute() -> dict[str, float]:
    return {
        "v2/SmileFrownLeft": 0.70,
        "v2/SmileFrownRight": 0.70,
        "v2/EyeSquintLeft": 0.78,
        "v2/EyeSquintRight": 0.78,
        "v2/EyeLidLeft": 0.50,
        "v2/EyeLidRight": 0.50,
        "v2/PupilDilation": 0.95,
        "v2/BrowExpressionLeft": 0.10,
        "v2/BrowExpressionRight": 0.10,
        "v2/JawOpen": 0.04,
    }


EXPRESSION_PRESETS = {
    "neutral": expression_neutral,
    "happy": expression_happy,
    "sad": expression_sad,
    "angry": expression_angry,
    "surprised": expression_surprised,
    "scared": expression_scared,
    "thinking": expression_thinking,
    "shy": expression_shy,
    "smirk": expression_smirk,
    "sleepy": expression_sleepy,
    "cute": expression_cute,
}

MANAGED_PARAMS = sorted({k for fn in EXPRESSION_PRESETS.values() for k in fn().keys()})


def select_expression(text: str) -> str:
    lower = text.lower()

    if "emotion:" in lower or "expr:" in lower or "expression:" in lower:
        for name in EXPRESSION_PRESETS:
            if name in lower:
                return name

    if any(w in lower for w in ["scared", "terrified", "panic", "shock", "shocked", "afraid"]):
        return "scared"
    if any(w in lower for w in ["shy", "embarrass", "embarrassed", "blush"]):
        return "shy"
    if any(w in lower for w in ["smirk", "sly", "hehe"]):
        return "smirk"
    if any(w in lower for w in ["sleepy", "tired", "zzz"]):
        return "sleepy"
    if any(w in lower for w in ["cute", "aww"]):
        return "cute"

    if any(word in lower for word in ["happy", "great", "awesome", "love", "thanks"]):
        return "happy"
    if any(word in lower for word in ["sad", "sorry", "unfortunate", "miss", "cry"]):
        return "sad"
    if any(word in lower for word in ["angry", "mad", "annoy", "hate", "furious"]):
        return "angry"
    if any(word in lower for word in ["wow", "surprise", "omg", "amazing"]):
        return "surprised"
    if "?" in text:
        return "thinking"
    return "neutral"


class OSCFaceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000

    param_root: str = "/avatar/parameters/"
    param_prefix: str = "FT"

    jaw_param: str = "v2/JawOpen"
    jaw_gain: float = 1.15
    jaw_min: float = 0.0
    jaw_max: float = 0.45
    jaw_smoothing: float = 0.38
    jaw_fps: int = 45
    jaw_curve: float = 0.65

    expression_strength: float = 1.0
    param_epsilon: float = 0.01
    reset_on_interrupt: bool = True

    chatbox_enabled: bool = True
    chatbox_send: bool = True
    chatbox_max_chars: int = 144
    text_flush_ms: int = 1200


class OSCFaceOutputs(TypedDict):
    pass


class OSCFace(
    Component[
        [Channel[TextFrame], Channel[AudioFrame], Channel[InterruptFrame]],
        OSCFaceOutputs,
    ]
):
    def __init__(self, config: OSCFaceConfig) -> None:
        super().__init__()
        self.config = config
        self._client = _OscClient(self.config.host, self.config.port)
        self._last_params: dict[str, float] = {}
        self._jaw_value: float | None = None
        self._jaw_last_send = 0.0
        self._text_lock = threading.Lock()
        self._text_buffer = ""
        self._last_text_time = 0.0

    def stop(self) -> None:
        self._client.close()
        super().stop()

    def get_output_channels(self) -> OSCFaceOutputs:
        return {}

    def _param_address(self, param: str) -> str:
        root = self.config.param_root
        if not root.startswith("/"):
            root = "/" + root
        if not root.endswith("/"):
            root += "/"

        prefix = self.config.param_prefix.strip("/")
        param = param.strip("/")

        if prefix:
            return f"{root}{prefix}/{param}"
        return f"{root}{param}"

    def _send_param(self, param: str, value: float) -> None:
        address = self._param_address(param)
        self._client.send_message(address, [float(value)])

    def _apply_expression(self, name: str) -> None:
        preset = EXPRESSION_PRESETS.get(name, expression_neutral)
        values = {k: 0.0 for k in MANAGED_PARAMS}
        values.update(preset())

        for param, raw_value in values.items():
            if param == self.config.jaw_param:
                continue

            value = float(raw_value) * self.config.expression_strength
            value = max(-1.0, min(1.0, value))

            prev = self._last_params.get(param)
            if prev is not None and abs(prev - value) < self.config.param_epsilon:
                continue

            self._send_param(param, value)
            self._last_params[param] = value

    def _send_chatbox(self, text: str) -> None:
        if not self.config.chatbox_enabled:
            return
        msg = text.strip()
        if not msg:
            return
        if self.config.chatbox_max_chars > 0:
            msg = msg[: self.config.chatbox_max_chars]
        self._client.send_message("/chatbox/input", [msg, self.config.chatbox_send])

    def _handle_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        expression = select_expression(cleaned)
        self._apply_expression(expression)
        self._send_chatbox(cleaned)

    def _flush_text_buffer(self) -> None:
        with self._text_lock:
            text = self._text_buffer
            self._text_buffer = ""
        if text:
            self._handle_text(text)

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
                self._handle_text(text)

    def _audio_loop(self, audio: Channel[AudioFrame]) -> None:
        for frame in audio.stream(self):
            if frame is None:
                break
            pcm = frame.get(data_format=AudioDataFormat.FLOAT32)
            if pcm is None:
                continue

            level = float(np.sqrt(np.mean(np.square(pcm))))

            raw = max(0.0, level * self.config.jaw_gain)
            curve = max(0.05, float(self.config.jaw_curve))
            shaped = raw**curve

            target = min(
                max(shaped, self.config.jaw_min),
                self.config.jaw_max,
            )

            if self._jaw_value is None:
                self._jaw_value = target
            else:
                s = self.config.jaw_smoothing
                self._jaw_value = (1.0 - s) * self._jaw_value + s * target

            now = time.monotonic()
            if now - self._jaw_last_send >= 1.0 / max(self.config.jaw_fps, 1):
                self._send_param(self.config.jaw_param, self._jaw_value)
                self._jaw_last_send = now

    def _interrupt_loop(self, interrupt: Channel[InterruptFrame]) -> None:
        for frame in interrupt.stream(self):
            if frame is None:
                break
            with self._text_lock:
                self._text_buffer = ""
            if self.config.reset_on_interrupt:
                self._apply_expression("neutral")

    def run(
            self,
            text: Channel[TextFrame] | None = None,
            audio: Channel[AudioFrame] | None = None,
            interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        threads: list[threading.Thread] = []

        if text is not None:
            threads.append(threading.Thread(target=self._text_loop, args=(text,)))
            threads.append(threading.Thread(target=self._text_flush_monitor))
        if audio is not None:
            threads.append(threading.Thread(target=self._audio_loop, args=(audio,)))
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