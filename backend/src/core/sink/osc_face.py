from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import requests
from pydantic import BaseModel
from pythonosc.udp_client import SimpleUDPClient

from src.core.channel import Receiver
from src.core.component import ThreadedComponent, Tag
from src.core.frames import InterruptFrame, TextFrame

GENERATE_END_FLAG = "[END_OF_GENERATE]"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_env_value(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip()
    quote_chars = "\"'‘’“”`"
    while cleaned and cleaned[0] in quote_chars:
        cleaned = cleaned[1:].lstrip()
    while cleaned and cleaned[-1] in quote_chars:
        cleaned = cleaned[:-1].rstrip()
    return cleaned


class _OscClient:
    def __init__(self, host: str, port: int) -> None:
        self._client = SimpleUDPClient(host, port)
        self._lock = threading.Lock()

    def send_message(self, address: str, value: Any) -> None:
        with self._lock:
            self._client.send_message(address, value)

    def close(self) -> None:
        return


class OSCFaceConfig(BaseModel):
    host: str | None = None
    port: int | None = None
    osc_prefix: str | None = None
    param_prefix: str | None = None

    hold_seconds: float | None = None
    transition_seconds: float | None = None
    transition_steps: int | None = None
    send_gap_seconds: float | None = None
    text_flush_ms: int | None = None

    emotion_llm_enable: bool | None = None
    emotion_llm_timeout_s: float | None = None
    emotion_llm_url: str | None = None
    emotion_llm_model: str | None = None
    emotion_llm_api_key_env_var: str = "GROQ_API_KEY"

    fusion_mode: str | None = None
    force_llm: bool | None = None
    debug_print_model_repr: bool | None = None


class OSCFaceInputs(NamedTuple):
    text: Receiver[TextFrame] | None = None
    interrupt: Receiver[InterruptFrame] | None = None


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def expression_neutral() -> Dict[str, float]:
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


def expression_happy() -> Dict[str, float]:
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


def expression_sad() -> Dict[str, float]:
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


def expression_angry() -> Dict[str, float]:
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


def expression_surprised() -> Dict[str, float]:
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


def expression_scared() -> Dict[str, float]:
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


def expression_thinking() -> Dict[str, float]:
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


def expression_shy() -> Dict[str, float]:
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


def expression_smirk() -> Dict[str, float]:
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


def expression_sleepy() -> Dict[str, float]:
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


def expression_cute() -> Dict[str, float]:
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
_ALLOWED_EXPRESSIONS = set(EXPRESSION_PRESETS.keys())

_LLM_SYSTEM_PROMPT = """
You are an emotion classifier for a VRChat avatar.

Given ONE user utterance, choose exactly ONE expression from this list:
neutral, happy, sad, angry, surprised, scared, thinking, shy, smirk, sleepy, cute

Return STRICT JSON only. No explanation. No markdown.
Schema:
{"expression": "<one of the listed expressions>"}
"""


def build_full_targets(
    base_neutral: Dict[str, float],
) -> Tuple[Dict[str, float], List[str]]:
    keys = list(MANAGED_PARAMS)
    base_defaults = {k: float(base_neutral.get(k, 0.0)) for k in keys}
    return base_defaults, keys


def preset_to_full(
    preset_name: str, base_defaults: Dict[str, float]
) -> Dict[str, float]:
    fn = EXPRESSION_PRESETS.get(preset_name, expression_neutral)
    full = dict(base_defaults)
    full.update({k: float(v) for k, v in fn().items()})
    return full


class OSCFace(ThreadedComponent[OSCFaceInputs, tuple[()]]):
    tags = Tag(io={"sink"}, functionality={"movement"})
    description = "Sends facial tracking data via OSC"

    def __init__(self, config: OSCFaceConfig | None = None) -> None:
        super().__init__()
        self.config = config or OSCFaceConfig()

        self.host = self.config.host or os.getenv("VRCHAT_IP") or "127.0.0.1"
        self.port = (
            int(self.config.port)
            if self.config.port is not None
            else int(os.getenv("VRCHAT_PORT", "9000"))
        )
        self.osc_prefix = self.config.osc_prefix or os.getenv(
            "OSC_PREFIX", "/avatar/parameters"
        )
        self.param_prefix = self.config.param_prefix or os.getenv(
            "VRCHAT_PARAM_PREFIX", "FT"
        )

        hold_seconds_value = (
            self.config.hold_seconds
            if self.config.hold_seconds is not None
            else float(os.getenv("HOLD_SECONDS", "2.0"))
        )
        self.hold_seconds = float(hold_seconds_value)
        transition_seconds_value = (
            self.config.transition_seconds
            if self.config.transition_seconds is not None
            else float(os.getenv("TRANSITION_SECONDS", "0.3"))
        )
        self.transition_seconds = float(transition_seconds_value)
        self.transition_steps = max(
            1,
            int(
                self.config.transition_steps
                if self.config.transition_steps is not None
                else int(os.getenv("TRANSITION_STEPS", "20"))
            ),
        )
        self.step_sleep = self.transition_seconds / self.transition_steps
        send_gap_seconds_value = (
            self.config.send_gap_seconds
            if self.config.send_gap_seconds is not None
            else float(os.getenv("SEND_GAP_SECONDS", "0.0"))
        )
        self.send_gap_seconds = float(send_gap_seconds_value)
        text_flush_ms_value = (
            self.config.text_flush_ms
            if self.config.text_flush_ms is not None
            else int(os.getenv("OSC_FACE_TEXT_FLUSH_MS", "600"))
        )
        self.text_flush_ms = int(text_flush_ms_value)

        self.emotion_llm_enable = (
            self.config.emotion_llm_enable
            if self.config.emotion_llm_enable is not None
            else _env_bool("EMOTION_LLM_ENABLE", False)
        )
        emotion_llm_timeout_value = (
            self.config.emotion_llm_timeout_s
            if self.config.emotion_llm_timeout_s is not None
            else float(os.getenv("EMOTION_LLM_TIMEOUT_S", "2.0"))
        )
        self.emotion_llm_timeout_s = float(emotion_llm_timeout_value)
        self.emotion_llm_url = (
            self.config.emotion_llm_url
            if self.config.emotion_llm_url is not None
            else os.getenv("EMOTION_LLM_URL", "")
        ).strip()
        self.emotion_llm_model = _sanitize_env_value(
            self.config.emotion_llm_model
            if self.config.emotion_llm_model is not None
            else os.getenv("EMOTION_LLM_MODEL", "moonshotai/kimi-k2-instruct-0905")
        )
        self.emotion_llm_api_key_env_var = (
            self.config.emotion_llm_api_key_env_var or "GROQ_API_KEY"
        )

        self.fusion_mode = (
            (
                self.config.fusion_mode
                if self.config.fusion_mode is not None
                else os.getenv("FUSION_MODE", "rule_first")
            )
            .strip()
            .lower()
        )
        if self.fusion_mode not in {"rule_first", "llm_first"}:
            self.fusion_mode = "rule_first"

        self.force_llm = (
            self.config.force_llm
            if self.config.force_llm is not None
            else _env_bool("FORCE_LLM", False)
        )
        self.debug_print_model_repr = (
            self.config.debug_print_model_repr
            if self.config.debug_print_model_repr is not None
            else _env_bool("DEBUG_PRINT_MODEL_REPR", True)
        )

        self._client = _OscClient(self.host, self.port)

        self._text_lock = threading.Lock()
        self._text_buffer = ""
        self._last_text_time = 0.0

        self._expr_lock = threading.Lock()
        self._expr_queue: deque[str] = deque()
        self._expr_event = threading.Event()

        self._base_defaults, self._keys = build_full_targets(expression_neutral())
        self._current_state = preset_to_full("neutral", self._base_defaults)

    def stop(self) -> None:
        self._client.close()
        super().stop()

    def _build_param_path(self, param_name: str) -> str:
        return f"{self.param_prefix}/{param_name}" if self.param_prefix else param_name

    def _send_params(self, params: Dict[str, float]) -> None:
        for name, value in params.items():
            address = f"{self.osc_prefix}/{self._build_param_path(name)}"
            self._client.send_message(address, float(value))
            if self.send_gap_seconds > 0:
                time.sleep(self.send_gap_seconds)

    def _transition(
        self, current: Dict[str, float], target: Dict[str, float]
    ) -> Dict[str, float]:
        for i in range(1, self.transition_steps + 1):
            t = i / self.transition_steps
            frame = {
                k: lerp(float(current.get(k, 0.0)), float(target.get(k, 0.0)), t)
                for k in self._keys
            }
            self._send_params(frame)
            time.sleep(self.step_sleep)
        return dict(target)

    def _select_expression_rule(self, text: str) -> Tuple[str, List[str], bool]:
        lower = (text or "").lower()
        hits: List[str] = []

        if any(tag in lower for tag in ["emotion:", "expr:", "expression:"]):
            for name in EXPRESSION_PRESETS.keys():
                if name in lower:
                    hits.append(f"manual:{name}")
                    return name, hits, True

        def any_hit(words: List[str]) -> bool:
            for word in words:
                if word in lower:
                    hits.append(word)
                    return True
            return False

        if any_hit(
            [
                "scared",
                "terrified",
                "panic",
                "shock",
                "shocked",
                "afraid",
                "fear",
                "frightened",
                "startled",
            ]
        ):
            return "scared", hits, True
        if any_hit(["shy", "embarrass", "embarrassed", "blush", "awkward", "blushing"]):
            return "shy", hits, True
        if any_hit(["smirk", "sly", "hehe", "mischievous", "sneer"]):
            return "smirk", hits, True
        if any_hit(["sleepy", "tired", "zzz", "want to sleep", "very sleepy"]):
            return "sleepy", hits, True
        if any_hit(["cute", "aww", "adorable", "awww"]):
            return "cute", hits, True
        if any_hit(
            [
                "happy",
                "great",
                "awesome",
                "love",
                "thanks",
                "glad",
                "yay",
                "thank you",
                "love you",
            ]
        ):
            return "happy", hits, True
        if any_hit(
            [
                "sad",
                "sorry",
                "unfortunate",
                "miss",
                "cry",
                "heartbroken",
                "apology",
            ]
        ):
            return "sad", hits, True
        if any_hit(
            ["angry", "mad", "annoy", "hate", "furious", "irritated", "dislike"]
        ):
            return "angry", hits, True
        if any_hit(
            [
                "wow",
                "surprise",
                "omg",
                "amazing",
                "shocked",
                "surprised",
                "oh my god",
                "wtf",
            ]
        ):
            return "surprised", hits, True

        if "?" in text:
            hits.append("?")
            return "thinking", hits, False

        return "neutral", hits, False

    def _select_expression_llm(self, text: str) -> Tuple[Optional[str], str]:
        if not self.emotion_llm_enable:
            return None, "LLM disabled (EMOTION_LLM_ENABLE!=1)"
        if not self.emotion_llm_url:
            return None, "Missing EMOTION_LLM_URL"

        api_key = os.getenv(self.emotion_llm_api_key_env_var, "")
        if not api_key:
            return None, f"Missing {self.emotion_llm_api_key_env_var}"

        payload = {
            "model": self.emotion_llm_model,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "max_tokens": 60,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.emotion_llm_url,
                json=payload,
                headers=headers,
                timeout=self.emotion_llm_timeout_s,
            )
            if response.status_code != 200:
                return None, f"LLM HTTP {response.status_code}: {response.text}"

            content = response.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            expr = str(data.get("expression", "")).strip().lower()
            if expr in _ALLOWED_EXPRESSIONS:
                return expr, f"LLM ok: {content}"
            return None, f"LLM invalid output: {content}"
        except Exception as exc:
            return None, f"LLM error: {exc!r}"

    def _fuse_expression(self, text: str) -> Tuple[str, Dict[str, str]]:
        rule_expr, hits, strong = self._select_expression_rule(text)

        dbg: Dict[str, str] = {
            "rule_expr": rule_expr,
            "rule_hits": ", ".join(hits) if hits else "(none)",
            "rule_strong": "1" if strong else "0",
            "fusion_mode": self.fusion_mode,
            "llm_enabled": "1" if self.emotion_llm_enable else "0",
            "force_llm": "1" if self.force_llm else "0",
        }

        llm_expr: Optional[str] = None
        llm_detail: str = "(not called)"

        def call_llm() -> None:
            nonlocal llm_expr, llm_detail
            llm_expr, llm_detail = self._select_expression_llm(text)
            dbg["llm_expr"] = llm_expr if llm_expr else "(none)"
            dbg["llm_detail"] = llm_detail

        if self.fusion_mode == "llm_first":
            call_llm()
            final_expr = llm_expr if llm_expr else rule_expr
            dbg["final_reason"] = "llm_first"
            return final_expr, dbg

        if self.force_llm or (
            self.emotion_llm_enable
            and (not strong)
            and rule_expr in ["neutral", "thinking"]
        ):
            call_llm()
            if llm_expr:
                dbg["final_reason"] = "rule_weak->llm_override"
                return llm_expr, dbg
            dbg["final_reason"] = "rule_weak->llm_failed_fallback_rule"
            return rule_expr, dbg

        dbg["llm_expr"] = "(not called)"
        dbg["llm_detail"] = "(not called)"
        dbg["final_reason"] = "rule_strong_or_llm_off"
        return rule_expr, dbg

    def _enqueue_text(self, text: str) -> None:
        content = text.strip()
        if not content:
            return
        with self._expr_lock:
            self._expr_queue.append(content)
            self._expr_event.set()

    def _flush_text_buffer(self) -> None:
        with self._text_lock:
            text = self._text_buffer
            self._text_buffer = ""
        if text:
            self._enqueue_text(text)

    def _text_loop(self, text_recv: Receiver[TextFrame]) -> None:
        for frame in text_recv:
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
        if self.text_flush_ms <= 0:
            return
        threshold = float(self.text_flush_ms) / 1000.0

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

    def _interrupt_loop(self, interrupt_recv: Receiver[InterruptFrame]) -> None:
        for frame in interrupt_recv:
            if frame is None:
                break

            with self._text_lock:
                self._text_buffer = ""

            with self._expr_lock:
                self._expr_queue.clear()
                self._expr_event.clear()

    def _expression_worker(self) -> None:
        while not self.stop_event.is_set():
            self._expr_event.wait(timeout=0.1)

            while True:
                with self._expr_lock:
                    if not self._expr_queue:
                        self._expr_event.clear()
                        break
                    text = self._expr_queue.popleft()

                expr, dbg = self._fuse_expression(text)
                if self.debug_print_model_repr:
                    print("[OSCFace DEBUG]")
                    print(f"  llm_text      : {text}")
                    print(f"  rule_expr     : {dbg.get('rule_expr')}")
                    print(f"  rule_hits     : {dbg.get('rule_hits')}")
                    print(f"  rule_strong   : {dbg.get('rule_strong')}")
                    print(f"  llm_expr      : {dbg.get('llm_expr')}")
                    print(f"  llm_detail    : {dbg.get('llm_detail')}")
                    print(f"  final_expr    : {expr}")
                    print(f"  final_reason  : {dbg.get('final_reason')}")
                    print("")

                target_state = preset_to_full(expr, self._base_defaults)
                self._current_state = self._transition(
                    self._current_state, target_state
                )
                if self.hold_seconds > 0:
                    time.sleep(self.hold_seconds)

    def run(self, inputs: OSCFaceInputs, outputs: tuple[()]) -> None:
        _ = outputs

        self._send_params(self._current_state)
        time.sleep(0.15)

        if self.debug_print_model_repr:
            print("[OSCFace CONFIG]")
            print(f"  target        = {self.host}:{self.port}")
            print(f"  osc_prefix    = {self.osc_prefix}")
            print(f"  param_prefix  = {self.param_prefix}")
            print(f"  llm_enable    = {self.emotion_llm_enable}")
            print(f"  llm_url_set   = {bool(self.emotion_llm_url)}")
            print(f"  llm_model     = {repr(self.emotion_llm_model)}")
            print(
                f"  llm_api_key   = {'(set)' if bool(os.getenv(self.emotion_llm_api_key_env_var, '')) else '(missing)'}"
            )
            print(f"  fusion_mode   = {self.fusion_mode}")
            print(f"  force_llm     = {self.force_llm}")
            print("")

        threads: list[threading.Thread] = []

        if inputs.text is not None:
            threads.append(
                threading.Thread(target=self._text_loop, args=(inputs.text,))
            )
            threads.append(threading.Thread(target=self._text_flush_monitor))
            threads.append(threading.Thread(target=self._expression_worker))

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
