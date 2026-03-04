from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import Component
from src.core.frames import TextFrame
from src.core.channel import UITextSender


class TextDisplayInputs(NamedTuple):
    text: Receiver[TextFrame]


class TextDisplayOutputs(NamedTuple):
    ui_text: UITextSender


class TextDisplay(Component[TextDisplayInputs, TextDisplayOutputs]):
    """Receives text frames and displays them in the node UI."""

    def run(self, inputs: TextDisplayInputs, outputs: TextDisplayOutputs) -> None:
        for frame in inputs.text(self):
            if frame is None:
                break
            outputs.ui_text.send(frame)
