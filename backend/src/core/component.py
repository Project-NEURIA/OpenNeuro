from __future__ import annotations

import inspect
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any, get_type_hints

from pydantic import BaseModel

from src.core.channel import ChannelSnapshot


class ComponentSnapshot(BaseModel):
    name: str
    status: str
    started_at: float | None
    channels: dict[str, ChannelSnapshot]


class Status(Enum):
    STARTUP = "startup"
    RUNNING = "running"
    STOPPED = "stopped"


class Component[**P, O: Mapping[str, Any]](ABC):
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
    def run(self, *args: P.args, **kwargs: P.kwargs) -> None: ...

    @abstractmethod
    def get_output_channels(self) -> O: ...

    def _safe_run(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self._status = Status.RUNNING
        self._started_at = time.time()
        try:
            self.run(*args, **kwargs)
        finally:
            self._status = Status.STOPPED

    def start(self, *args: P.args, **kwargs: P.kwargs) -> None:
        if self.status == Status.RUNNING:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._safe_run, args=args, kwargs=kwargs, daemon=True)
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
            channels={n: ch.snapshot() for n, ch in self.get_output_channels().items()},
        )

    @classmethod
    def get_init_types(cls) -> dict[str, type]:
        """Returns {param_name: type} from __init__, excluding self."""
        hints = get_type_hints(cls.__init__)
        hints.pop("return", None)
        return hints

    @classmethod
    def get_input_types(cls) -> dict[str, type]:
        """Introspect run()'s keyword params for input channel types."""
        hints = get_type_hints(cls.run)
        hints.pop("return", None)
        return hints

    @classmethod
    def get_output_types(cls) -> dict[str, type]:
        """Introspect get_output_channels()'s return type (TypedDict) for output types."""
        hints = get_type_hints(cls.get_output_channels)
        td = hints.get("return")
        if td is None:
            return {}
        return get_type_hints(td)

    @classmethod
    def registered_subclasses(cls) -> dict[str, type[Component[..., Any]]]:
        """Returns all concrete subclasses as {name: class}, walking the full hierarchy."""
        from src.core import source, sink, conduit  # noqa: F401

        result: dict[str, type[Component[..., Any]]] = {}

        def walk(subclass: type[Component[..., Any]]) -> None:
            if not inspect.isabstract(subclass):
                result[subclass.__name__] = subclass
            for child in subclass.__subclasses__():
                walk(child)

        for child in cls.__subclasses__():
            walk(child)

        return result
