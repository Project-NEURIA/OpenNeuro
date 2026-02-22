from __future__ import annotations

import inspect
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from src.core.channel import Receiver, Sender


class Status(Enum):
    STARTUP = "startup"
    RUNNING = "running"
    STOPPED = "stopped"


class Component[I: tuple[Receiver[Any] | None, ...], O: tuple[Sender[Any] | None, ...]](
    ABC
):
    def __init__(self) -> None:
        self.name: str = type(self).__name__
        self._status = Status.STARTUP
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def status(self) -> Status:
        return self._status

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @abstractmethod
    def run(self, inputs: I, outputs: O) -> None: ...

    def _safe_run(self, inputs: I, outputs: O) -> None:
        self._status = Status.RUNNING
        try:
            self.run(inputs, outputs)
        finally:
            self._status = Status.STOPPED

    def start(self, inputs: I, outputs: O) -> None:
        if self.status == Status.RUNNING:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._safe_run, args=(inputs, outputs), daemon=True
        )
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

    @classmethod
    def get_init_types(cls) -> dict[str, type]:
        """Returns {param_name: type} from __init__, excluding self."""
        hints = get_type_hints(cls.__init__)
        hints.pop("return", None)
        return hints

    @classmethod
    def _get_type_param(cls, index: int) -> type | None:
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Component:
                args = get_args(base)
                if len(args) > index:
                    return args[index]
        return None

    @classmethod
    def _resolve_tuple_types(cls, tp: type | None) -> dict[str, type]:
        if tp is None:
            return {}
        if hasattr(tp, "_fields"):
            hints = get_type_hints(tp)
            return {name: hints[name] for name in tp._fields}
        args = get_args(tp)
        if not args or args == ((),):
            return {}
        return {str(i + 1): arg for i, arg in enumerate(args)}

    @classmethod
    def get_input_types(cls) -> dict[str, type]:
        return cls._resolve_tuple_types(cls._get_type_param(0))

    @classmethod
    def get_output_types(cls) -> dict[str, type]:
        return cls._resolve_tuple_types(cls._get_type_param(1))

    @classmethod
    def from_args(cls, init_args: dict[str, Any]) -> Component[Any, Any]:
        """Construct a component instance, deserializing any BaseModel init params."""
        kwargs: dict[str, Any] = {}
        for k, v in cls.get_init_types().items():
            if k not in init_args:
                continue
            if isinstance(v, type) and issubclass(v, BaseModel):
                kwargs[k] = v(**init_args[k])
            else:
                kwargs[k] = init_args[k]
        return cls(**kwargs) if kwargs else cls()  # type: ignore[call-arg]

    @classmethod
    def registered_subclasses(cls) -> dict[str, type[Component[Any, Any]]]:
        """Returns all concrete subclasses as {name: class}, walking the full hierarchy."""
        from src.core import source, sink, conduit  # noqa: F401

        result: dict[str, type[Component[Any, Any]]] = {}

        def walk(subclass: type[Component[Any, Any]]) -> None:
            if not inspect.isabstract(subclass):
                result[subclass.__name__] = subclass
            for child in subclass.__subclasses__():
                walk(child)

        for child in cls.__subclasses__():
            walk(child)

        return result
