"""Throttle conduit — rate-limits a stream by grabbing the newest frame at a fixed interval."""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag


class ThrottleConfig(BaseModel):
    interval: float = 0.4
    """Seconds between each output. Set higher than the slowest upstream to keep parallel streams in sync."""


class ThrottleInputs[T](NamedTuple):
    data: Receiver[T]


class ThrottleOutputs[T](NamedTuple):
    data: Sender[T]


class Throttle[T](ThreadedComponent[ThrottleInputs[T], ThrottleOutputs[T]]):
    description = (
        "Rate-limits a stream to a fixed interval, always forwarding the newest frame"
    )
    tags = Tag(io={"conduit"}, functionality={"misc"})

    def __init__(self, config: ThrottleConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, inputs: ThrottleInputs[T], outputs: ThrottleOutputs[T]) -> None:
        inputs.data.newest = True
        for item in inputs.data:
            if item is None:
                break
            outputs.data.send(item)
            if self.stop_event.wait(self.config.interval):
                break
