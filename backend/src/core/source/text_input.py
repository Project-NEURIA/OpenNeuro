from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import TextFrame
from src.core.channel import UITextReceiver


class TextInputInputs(NamedTuple):
    ui_text: UITextReceiver


class TextInputOutputs(NamedTuple):
    text: Sender[TextFrame]


class TextInput(ThreadedComponent[TextInputInputs, TextInputOutputs]):
    description = "Accepts text input from the UI"

    """Receives text from the frontend node UI and sends it downstream."""

    tags = Tag(io={"source"}, functionality={"misc"})

    def run(self, inputs: TextInputInputs, outputs: TextInputOutputs) -> None:
        for frame in inputs.ui_text(self):
            if frame is None:
                break
            outputs.text.send(frame)
