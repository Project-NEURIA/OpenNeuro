from __future__ import annotations

import inspect
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, get_type_hints

from pydantic import BaseModel

from openneuro.core.channel import Channel, ChannelSnapshot


class ComponentSnapshot(BaseModel):
    name: str
    status: str
    started_at: float | None
    channels: list[ChannelSnapshot]


class Status(Enum):
    STARTUP = "startup"
    RUNNING = "running"
    STOPPED = "stopped"


class Component[*ITs](ABC):
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

    @abstractmethod
    def get_output_channels(self) -> tuple[Channel, ...]:
        ...

    @abstractmethod
    def set_input_channels(self, *input_channels: *ITs) -> None:
        ...

    @abstractmethod
    def run(self) -> None:
        ...

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
        and expect the run() method to return. Input streams are unregistered from input channels as streams
        will periodically check and raise StopIteration when self._stop_event is set.
        """
        if self.status == Status.STOPPED:
            return
        self._stop_event.set()

    def snapshot(self) -> ComponentSnapshot:
        return ComponentSnapshot(
            name=self.name,
            status=self.status.value,
            started_at=self._started_at,
            channels=[t.snapshot() for t in self.get_output_channels()],
        )

    @classmethod
    def get_init_types(cls) -> dict[str, type]:
        """Returns {param_name: type} from __init__, excluding self."""
        hints = get_type_hints(cls.__init__)
        hints.pop("return", None)
        return hints

    @classmethod
    def get_input_types(cls) -> tuple[type, ...]:
        """Returns input types from the generic args Component[Channel[T1], Channel[T2], ...].

        Unwraps Channel[T] -> T.
        """
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is Component:
                args = getattr(base, "__args__", ())
                return tuple(
                    a.__args__[0]
                    for a in args
                    if getattr(a, "__args__", ())
                )
        return ()

    @classmethod
    def get_output_types(cls) -> tuple[type, ...]:
        """Returns output types from the return annotation of get_output_channels().

        e.g. tuple[Channel[bytes], Channel[str]] -> (bytes, str)
        """
        hints = get_type_hints(cls.get_output_channels)
        ret = hints.get("return")
        if ret is None:
            return ()
        return tuple(
            a.__args__[0]
            for a in getattr(ret, "__args__", ())
            if getattr(a, "__args__", ())
        )

    @classmethod
    def get_input_names(cls) -> list[str]:
        """Returns human-readable names for each input slot.
        
        Components can override this to provide descriptive labels.
        Default is generic names like 'input_0', 'input_1', etc.
        """
        num_inputs = len(cls.get_input_types())
        return [f"input_{i}" for i in range(num_inputs)]

    @classmethod
    def registered_subclasses(cls) -> dict[str, type[Component]]:
        """Returns all concrete subclasses as {name: class}, walking the full hierarchy."""
        from openneuro.core import source, sink, conduit  # noqa: F401

        result: dict[str, type[Component]] = {}

        def walk(subclass: type[Component[*tuple[Any, ...]]]) -> None:
            if not inspect.isabstract(subclass):
                result[subclass.__name__] = subclass
            for child in subclass.__subclasses__():
                walk(child)

        for child in cls.__subclasses__():
            walk(child)

        return result
