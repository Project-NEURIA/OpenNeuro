from __future__ import annotations

import inspect
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Never

from pydantic import BaseModel

from .topic import NOTOPIC, Topic


class ComponentMetadata(BaseModel):
    name: str
    status: str
    started_at: float | None

class Status(Enum):
    STARTUP = "startup"
    RUNNING = "running"
    STOPPED = "stopped"

class Component[I1 = Never, O1 = Never, I2 = Never, O2 = Never, I3 = Never, O3 = Never, I4 = Never, O4 = Never](ABC):
    def __init__(self) -> None:
        self.name: str = type(self).__name__
        self._status = Status.STARTUP
        self._started_at: float | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def status(self) -> Status:
        return self._status

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def get_output_topics(self) -> tuple[Topic[O1], Topic[O2], Topic[O3], Topic[O4]]:
        raise NotImplementedError

    def set_input_topics(
        self,
        t1: Topic[I1],
        t2: Topic[I2],
        t3: Topic[I3],
        t4: Topic[I4],
    ) -> None:
        raise NotImplementedError

    def metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name=self.name,
            status=self.status.value,
            started_at=self._started_at,
        )

    @abstractmethod
    def run(self) -> None: ...

    def _safe_run(self) -> None:
        self._status = Status.RUNNING
        self._started_at = time.time()
        try:
            self.run()
        finally:
            self._status = Status.STOPPED

    def start(self) -> None:
        if self.status == Status.RUNNING:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._safe_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Idempotent.
        When a component instance is stopped, it will stop the running thread cooperatively by setting self._stop_event
        and expect the run() method to return. Input streams are unregistered from input topics as streams
        will periodically check and raise StopIteration when self._stop_event is set.
        """
        if self.status == Status.STOPPED:
            return
        self._stop_event.set()

    @classmethod
    def get_sig(cls) -> tuple[tuple[type, ...], tuple[type, ...]]:
        """Returns (input_types, output_types) from the generic args.

        Generic params are interleaved: Component[I1, O1, I2, O2, ...].
        Even indices (0, 2, 4, 6) are input types, odd (1, 3, 5, 7) are output types.
        Never-typed params are filtered out.
        """
        input_types: tuple[type, ...] = ()
        output_types: tuple[type, ...] = ()
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is Component:
                args = getattr(base, "__args__", ())
                input_types = tuple(a for a in args[0::2] if a is not Never)
                output_types = tuple(a for a in args[1::2] if a is not Never)
                break

        return input_types, output_types

    @classmethod
    def registered_subclasses(cls) -> dict[str, type[Component]]:
        """Returns all concrete subclasses as {name: class}, walking the full hierarchy."""
        from .source import Mic
        from .sink import Speaker
        from .conduit import VAD, ASR, LLM, TTS, STS

        result: dict[str, type[Component]] = {}

        def walk(subclass: type[Component[Any, Any, Any, Any, Any, Any, Any, Any]]) -> None:
            if not inspect.isabstract(subclass):
                result[subclass.__name__] = subclass
            for child in subclass.__subclasses__():
                walk(child)

        for child in cls.__subclasses__():
            walk(child)

        return result