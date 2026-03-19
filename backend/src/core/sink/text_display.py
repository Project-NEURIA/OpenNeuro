from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import PrimitiveComponent, Tag
from src.core.frames import TextFrame
from src.core.channel import UITextSender


class TextDisplayInputs(NamedTuple):
    text: Receiver[TextFrame]


class TextDisplayOutputs(NamedTuple):
    ui_text: UITextSender


class TextDisplay(PrimitiveComponent[TextDisplayInputs, TextDisplayOutputs]):
    description = "Displays text output in the UI"

    """Receives text frames and displays them in the node UI."""

    _tags = Tag(io={"sink"}, functionality={"misc"})

    def run(self, inputs: TextDisplayInputs, outputs: TextDisplayOutputs) -> None:
        for frame in inputs.text(self):
            if frame is None:
                break
            outputs.ui_text.send(frame)
