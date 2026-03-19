from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag


class PassthroughInputs[T](NamedTuple):
    data: Receiver[T]


class PassthroughOutputs[T](NamedTuple):
    data: Sender[T]


class Passthrough[T](ThreadedComponent[PassthroughInputs[T], PassthroughOutputs[T]]):
    tags = Tag(io={"conduit"}, functionality={"misc"})
    description = "Passes data through unchanged"

    def run(self, inputs: PassthroughInputs[T], outputs: PassthroughOutputs[T]) -> None:
        for item in inputs.data(self):
            if item is None:
                break
            outputs.data.send(item)
