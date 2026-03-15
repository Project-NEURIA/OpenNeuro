from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent, Tag


class PassthroughInputs[T](NamedTuple):
    data: Receiver[T]


class PassthroughOutputs[T](NamedTuple):
    data: Sender[T]


class Passthrough[T](PrimitiveComponent[PassthroughInputs[T], PassthroughOutputs[T]]):
    _tags = Tag(io={"conduit"}, functionality={"misc"})

    def run(self, inputs: PassthroughInputs[T], outputs: PassthroughOutputs[T]) -> None:
        for item in inputs.data(self):
            if item is None:
                break
            outputs.data.send(item)
