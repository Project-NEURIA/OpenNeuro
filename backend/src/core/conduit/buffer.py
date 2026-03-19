from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import EOS


class BufferInputs[T](NamedTuple):
    data: Receiver[T | EOS]


class BufferOutputs[T](NamedTuple):
    batch: Sender[list[T]]


class Buffer[T](ThreadedComponent[BufferInputs[T], BufferOutputs[T]]):
    tags = Tag(io={"conduit"}, functionality={"misc"})
    description = "Buffers incoming data before forwarding"

    def run(self, inputs: BufferInputs[T], outputs: BufferOutputs[T]) -> None:
        buf: list[object] = []
        for item in inputs.data(self):
            if item is None:
                break
            if item is EOS.END:
                if buf:
                    outputs.batch.send(buf)  # type: ignore[arg-type]
                    buf = []
                continue
            buf.append(item)
