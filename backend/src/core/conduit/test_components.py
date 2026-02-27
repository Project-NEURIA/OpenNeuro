"""Test conduits with complex type signatures for type checker validation."""

from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import AudioFrame, Frame, TextFrame


# --- 1. Accepts base Frame, outputs AudioFrame ---
# Tests: concrete subtyping (AudioFrame ≤ Frame should be valid)


class FrameFilterInputs(NamedTuple):
    frames: Receiver[Frame]


class FrameFilterOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class FrameFilter(Component[FrameFilterInputs, FrameFilterOutputs]):
    def run(self, inputs: FrameFilterInputs, outputs: FrameFilterOutputs) -> None:
        for item in inputs.frames(self):
            if item is None:
                break
            if isinstance(item, AudioFrame):
                outputs.audio.send(item)


# --- 2. list[T] constructor types ---
# Tests: constructor decomposition (list[AudioFrame] ≤ list[Frame])


class BatchInputs(NamedTuple):
    items: Receiver[list[AudioFrame]]


class BatchOutputs(NamedTuple):
    batched: Sender[list[Frame]]


class Batcher(Component[BatchInputs, BatchOutputs]):
    def run(self, inputs: BatchInputs, outputs: BatchOutputs) -> None:
        for batch in inputs.items(self):
            if batch is None:
                break
            outputs.batched.send(list(batch))  # type: ignore[arg-type]


# --- 3. Union input type (non-optional) ---
# Tests: union types in slot declarations


class MergeInputs(NamedTuple):
    stream: Receiver[AudioFrame | TextFrame]


class MergeOutputs(NamedTuple):
    audio: Sender[AudioFrame]
    text: Sender[TextFrame]


class Demux(Component[MergeInputs, MergeOutputs]):
    def run(self, inputs: MergeInputs, outputs: MergeOutputs) -> None:
        for item in inputs.stream(self):
            if item is None:
                break
            if isinstance(item, AudioFrame):
                outputs.audio.send(item)
            elif isinstance(item, TextFrame):
                outputs.text.send(item)


# --- 4. Outputs a union ---
# Tests: union on output side


class MuxInputs(NamedTuple):
    audio: Receiver[AudioFrame]
    text: Receiver[TextFrame]


class MuxOutputs(NamedTuple):
    stream: Sender[AudioFrame | TextFrame]


class Mux(Component[MuxInputs, MuxOutputs]):
    def run(self, inputs: MuxInputs, outputs: MuxOutputs) -> None:
        for item in inputs.audio(self):
            if item is None:
                break
            outputs.stream.send(item)


# --- 5. Type variable passthrough ---
# Tests: T stays as a type var, gets unified through edges


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


# --- 6. Type variable in constructor ---
# Tests: list[T] where T is a type var


class BufferInputs[T](NamedTuple):
    data: Receiver[T]


class BufferOutputs[T](NamedTuple):
    batch: Sender[list[T]]


class Buffer[T](Component[BufferInputs[T], BufferOutputs[T]]):
    def run(self, inputs: BufferInputs[T], outputs: BufferOutputs[T]) -> None:
        buf: list[object] = []
        for item in inputs.data(self):
            if item is None:
                break
            buf.append(item)
            if len(buf) >= 10:
                outputs.batch.send(buf)  # type: ignore[arg-type]
                buf = []
