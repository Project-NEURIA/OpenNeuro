from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent
from src.core.frames import MessagesFrame, TextFrame


class MessagesToTextInputs(NamedTuple):
    messages: Receiver[MessagesFrame]


class MessagesToTextOutputs(NamedTuple):
    text: Sender[TextFrame]


class MessagesToText(PrimitiveComponent[MessagesToTextInputs, MessagesToTextOutputs]):
    """Converts MessagesFrame to TextFrame for display purposes."""

    def run(self, inputs: MessagesToTextInputs, outputs: MessagesToTextOutputs) -> None:
        for frame in inputs.messages(self):
            if frame is None:
                break
            # Use the pre-built text representation from MessagesFrame
            outputs.text.send(TextFrame.new(text=frame.text))
