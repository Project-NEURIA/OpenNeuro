from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import ThreadedComponent, Tag


class DoNothingInputs[T](NamedTuple):
    input: Receiver[T]


class DoNothing[T](ThreadedComponent[DoNothingInputs[T], tuple[()]]):
    tags = Tag(io={"sink"}, functionality={"misc"})
    description = "Consumes input and discards it"

    def run(self, inputs: DoNothingInputs[T], outputs: tuple[()]) -> None:
        for _ in inputs.input:
            pass
