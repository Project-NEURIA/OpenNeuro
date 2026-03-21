from __future__ import annotations

import time
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import RequestFrame


class PulseConfig(BaseModel):
    interval: float = 1.0


class PulseOutputs(NamedTuple):
    request: Sender[RequestFrame]


class Pulse(ThreadedComponent[tuple[()], PulseOutputs]):
    description = "Emits a request frame at a fixed interval"
    tags = Tag(io={"source"}, functionality={"misc"})

    def __init__(self, config: PulseConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, inputs: tuple[()], outputs: PulseOutputs) -> None:
        while not self.stop_event.is_set():
            outputs.request.send(RequestFrame.new())
            self.stop_event.wait(self.config.interval)
