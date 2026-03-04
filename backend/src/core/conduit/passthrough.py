from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import Component


class PassthroughInputs[T](NamedTuple):
    data: Receiver[T]


class PassthroughOutputs[T](NamedTuple):
    data: Sender[T]


class Passthrough[T](Component[PassthroughInputs[T], PassthroughOutputs[T]]):
    def run(self, inputs: PassthroughInputs[T], outputs: PassthroughOutputs[T]) -> None:
        for item in inputs.data(self):
            if item is None:
                break
            outputs.data.send(item)
