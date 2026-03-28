from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import ThreadedComponent, Tag
from src.core.frames import TextFrame
from src.core.channel import UITextSender


class TextDisplayInputs(NamedTuple):
    text: Receiver[TextFrame]


class TextDisplayOutputs(NamedTuple):
    ui_text: UITextSender


class TextDisplay(ThreadedComponent[TextDisplayInputs, TextDisplayOutputs]):
    description = "Displays **text output** in the node UI widget. Receives `TextFrame` input and renders it as a live text stream in the frontend."

    """Receives text frames and displays them in the node UI."""

    tags = Tag(io={"sink"}, functionality={"misc"})

    def run(self, inputs: TextDisplayInputs, outputs: TextDisplayOutputs) -> None:
        for frame in inputs.text:
            if frame is None:
                break
            outputs.ui_text.send(frame)
