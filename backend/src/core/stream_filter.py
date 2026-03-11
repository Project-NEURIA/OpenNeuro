from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


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
