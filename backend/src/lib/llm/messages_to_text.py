from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import MessageFrame, TextFrame


class MessagesToTextInputs(NamedTuple):
    messages: Receiver[list[MessageFrame]]


class MessagesToTextOutputs(NamedTuple):
    text: Sender[TextFrame]


class MessagesToText(ThreadedComponent[MessagesToTextInputs, MessagesToTextOutputs]):
    description = "Converts message objects to plain text"

    """Converts list[MessageFrame] to TextFrame for display purposes."""

    tags = Tag(io={"conduit"}, functionality={"llm"})

    def run(self, inputs: MessagesToTextInputs, outputs: MessagesToTextOutputs) -> None:
        for frame in inputs.messages:
            if frame is None:
                break
            text = "\n".join(f"{m.role}: {m.content}" for m in frame)
            outputs.text.send(TextFrame.new(text=text))
