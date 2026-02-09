from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Never, overload

from .topic import Topic

class Status(Enum):
    STARTUP = "startup"
    RUNNING = "running"
    STOPPED = "stopped"


_NONE: Topic[Never] = Topic(name="_none")
# Remove _NONE from the registry — it's internal

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
        padded = topics + (_NONE,) * (4 - len(topics))
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
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def topic(self) -> Topic[O1]:
        return self._topics[0]

    def topics(self) -> tuple[Topic[O1], Topic[O2], Topic[O3], Topic[O4]]:
        return self._topics

    def set_input_topics(self, *topics: Topic) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def _safe_run(self) -> None:
        self._status = Status.RUNNING
        self._started_at = time.time()
        try:
            self.run()
        finally:
            self._status = Status.STOPPED

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "status": self._status,
            "started_at": self._started_at,
            "error": self._error,
        }

    def start(self) -> None:
        if self._status == Status.RUNNING:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._safe_run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """
        Idempotent.
        When a node instance is paused, it will stop the running thread cooperatively by setting self._stop_event
        and expect the run() method to return. Input streams are unregistered from input topics as streams
        will periodically check and raise StopIteration when self._stop_event is set.
        This method does not do anything to the output topics, since the node instance is only paused, not killed.
        """
        if self._status == Status.STOPPED:
            return
        self._stop_event.set()
