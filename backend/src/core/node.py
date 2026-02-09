from __future__ import annotations

import inspect
import threading
import time
from enum import Enum
from typing import Any, Never, get_type_hints, overload

from pydantic import BaseModel

from .topic import Topic


class NodeMetadata(BaseModel):
    name: str
    status: str
    started_at: float | None

class Status(Enum):
    STARTUP = "startup"
    RUNNING = "running"
    STOPPED = "stopped"

class Node[O1, O2 = Never, O3 = Never, O4 = Never]:

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, c1: Topic[O1], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[O1], c2: Topic[O2], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[O1], c2: Topic[O2], c3: Topic[O3], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[O1], c2: Topic[O2], c3: Topic[O3], c4: Topic[O4], /) -> None: ...

    def __init__(self, *topics: Topic) -> None:  # type: ignore[override]
        padded = topics + tuple([Topic[Never] for _ in range(4 - len(topics))])
        self._topics: tuple[Topic[O1], Topic[O2], Topic[O3], Topic[O4]] = padded  # type: ignore[assignment]
        self.name: str = type(self).__name__
        self._status = Status.STARTUP
        self._started_at: float | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def status(self) -> Status:
        return self._status

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def topic(self) -> Topic[O1]:
        return self._topics[0]

    def topics(self) -> tuple[Topic[O1], Topic[O2], Topic[O3], Topic[O4]]:
        return self._topics

    def set_input_topics(self, *topics: Topic) -> None:
        pass

    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name=self.name,
            status=self.status.value,
            started_at=self._started_at,
        )

    def run(self) -> None:
        raise NotImplementedError

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
        When a node instance is stopped, it will stop the running thread cooperatively by setting self._stop_event
        and expect the run() method to return. Input streams are unregistered from input topics as streams
        will periodically check and raise StopIteration when self._stop_event is set.
        """
        if self.status == Status.STOPPED:
            return
        self._stop_event.set()

    @classmethod
    def get_sig(cls) -> tuple[dict[str, type], tuple[type, ...]]:
        """Returns (input_params, output_types).

        input_params: __init__ param names to types (excluding self).
        output_types: generic args from the class definition (e.g. Node[bytes] -> (bytes,)).
        """
        sig = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__)

        input_params: dict[str, type] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            t = hints.get(name, param.annotation)
            if t is inspect._empty:
                t = Any
            input_params[name] = t

        output_types: tuple[type, ...] = ()
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is Node:
                args = getattr(base, "__args__", ())
                output_types = tuple(a for a in args if a is not Never)
                break

        return input_params, output_types

    @classmethod
    def registered_subclasses(cls) -> dict[str, type[Node]]:
        """Returns all concrete subclasses as {name: class}, walking the full hierarchy."""
        from .source import Mic
        from .sink import Speaker
        from .conduit import VAD, ASR, LLM, TTS, STS

        result: dict[str, type[Node]] = {}

        def walk(subclass: type[Node]) -> None:
            if not inspect.isabstract(subclass):
                result[subclass.__name__] = subclass
            for child in subclass.__subclasses__():
                walk(child)

        for child in cls.__subclasses__():
            walk(child)

        return result