"""PromptRepeater source — emits a static TextFrame on a fixed interval."""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Sender
from src.core.component import PrimitiveComponent
from src.core.frames import TextFrame


class PromptRepeaterConfig(BaseModel):
    prompt: str = "hand, head"
    interval_seconds: float = 3.0


class PromptRepeaterOutputs(NamedTuple):
    text: Sender[TextFrame]


class PromptRepeater(PrimitiveComponent[tuple[()], PromptRepeaterOutputs]):
    def __init__(self, config: PromptRepeaterConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, inputs: tuple[()], outputs: PromptRepeaterOutputs) -> None:
        print(
            f"[PromptRepeater] Emitting '{self.config.prompt}' every {self.config.interval_seconds}s"
        )
        while not self.stop_event.is_set():
            outputs.text.send(TextFrame.new(text=self.config.prompt))
            self.stop_event.wait(self.config.interval_seconds)
        print("[PromptRepeater] Stopped")
