from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver
from src.core.component import PrimitiveComponent


class DoNothingInputs[T](NamedTuple):
    input: Receiver[T]


class DoNothing[T](PrimitiveComponent[DoNothingInputs[T], tuple[()]]):
    def run(self, inputs: DoNothingInputs[T], outputs: tuple[()]) -> None:
        for _ in inputs.input(self):
            pass
