from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypedDict

from openai import OpenAI

from src.core.component import Component
from src.core.channel import Channel

CutFn = Callable[[str], int]


def _cut_space(buf: str) -> int:
    return max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))


_SENT_END = re.compile(r"""[.!?…]+["'\)\]\}]*($|\s)""")


def _cut_sentence(buf: str) -> int:
    buf.strip()
    last = -1
    for m in _SENT_END.finditer(buf):
        cut = m.end() - 1
        while cut >= 0 and buf[cut].isspace():
            cut -= 1
        last = max(last, cut)
    return last


class StreamFilter:
    speak_buf: str
    in_square: int
    in_paren: int
    in_angle: int
    in_bold: bool
    in_italic: bool
    cut_fn: CutFn

    def __init__(self) -> None:
        self.speak_buf = ""
        self.in_square = 0
        self.in_paren = 0
        self.in_angle = 0
        self.in_bold = False
        self.in_italic = False
        self.cut_fn = _cut_sentence

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
                    self.in_square += 1; i += 1; continue
                if ch == "]" and self.in_square:
                    self.in_square -= 1; i += 1; continue
                if ch == "(":
                    self.in_paren += 1; i += 1; continue
                if ch == ")" and self.in_paren:
                    self.in_paren -= 1; i += 1; continue
                if ch == "<":
                    self.in_angle += 1; i += 1; continue
                if ch == ">" and self.in_angle:
                    self.in_angle -= 1; i += 1; continue

            if self.in_square or self.in_paren or self.in_angle or self.in_bold or self.in_italic:
                i += 1
                continue

            self.speak_buf += ch
            i += 1


class TTSOutputs(TypedDict):
    audio: Channel[bytes]


class TTS(Component[[Channel[str]], TTSOutputs]):
    def __init__(self) -> None:
        super().__init__()
        self._client = OpenAI()
        self._filter = StreamFilter()
        self._output_audio: Channel[bytes] = Channel(name="audio")

    def get_output_channels(self) -> TTSOutputs:
        return {"audio": self._output_audio}

    def run(self, text: Channel[str]) -> None:
        for chunk in text.stream(self):
            if chunk is None:
                break
            if chunk == "":
                t = self._filter.feed("", force=True)
            else:
                t = self._filter.feed(chunk)

            if t and t.strip():
                self._synthesize(t)

    def _synthesize(self, text: str) -> None:
        try:
            with self._client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="pcm",
            ) as response:
                for pcm in response.iter_bytes(chunk_size=4096):
                    self._output_audio.send(pcm)
        except Exception as e:
            print(f"[TTS] Error: {e}")
