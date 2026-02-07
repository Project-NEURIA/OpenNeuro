from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable

import requests

from ..node import Node
from ..topic import Topic, Stream

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


class TTS(Node[bytes]):
    def __init__(
        self,
        source: Stream[str],
        *,
        url: str,
        voice_id: str,
        model_id: str,
        credential: str = "",
    ) -> None:
        self._source = source
        self._url = url
        self._voice_id = voice_id
        self._model_id = model_id
        self._credential = credential
        self._filter = StreamFilter()
        self.output = Topic[bytes]()
        super().__init__(self.output)

    def run(self) -> None:
        for chunk in self._source:
            if chunk == "":
                text = self._filter.feed("", force=True)
            else:
                text = self._filter.feed(chunk)

            if text and text.strip():
                self._synthesize(text)

    def _synthesize(self, text: str) -> None:
        headers = {
            "Authorization": f"Basic {self._credential}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voiceId": self._voice_id,
            "modelId": self._model_id,
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000},
        }
        try:
            r = requests.post(self._url, json=payload, headers=headers, stream=True, timeout=10)
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                raw = base64.b64decode(msg["result"]["audioContent"])
                if len(raw) > 44:
                    pcm = raw[44:]  # strip WAV header
                    self.output.send(pcm)
        except Exception as e:
            print(f"[TTS] Error: {e}")
