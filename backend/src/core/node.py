from __future__ import annotations

import threading
import time
import traceback
from typing import Never, overload

from .topic import Topic

_NONE: Topic[Never] = Topic(name="_none")
# Remove _NONE from the registry — it's internal
Topic._registry.remove(_NONE)


class Node[T1, T2 = Never, T3 = Never, T4 = Never]:
    _registry: list = []

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, c1: Topic[T1], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[T1], c2: Topic[T2], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[T1], c2: Topic[T2], c3: Topic[T3], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[T1], c2: Topic[T2], c3: Topic[T3], c4: Topic[T4], /) -> None: ...

    def __init__(self, *topics: Topic) -> None:  # type: ignore[override]
        padded = topics + (_NONE,) * (4 - len(topics))
        self._topics: tuple[Topic[T1], Topic[T2], Topic[T3], Topic[T4]] = padded  # type: ignore[assignment]
        self.name: str = type(self).__name__
        self._status: str = "idle"
        self._started_at: float | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        Node._registry.append(self)

    @property
    def topic(self) -> Topic[T1]:
        return self._topics[0]

    def get_topics(self) -> tuple[Topic[T1], Topic[T2], Topic[T3], Topic[T4]]:
        return self._topics

    def run(self) -> None:
        raise NotImplementedError

    def _safe_run(self) -> None:
        self._status = "running"
        self._started_at = time.time()
        try:
            self.run()
        except StopIteration:
            if self._status == "running":
                self._status = "stopped"
        except Exception as e:
            self._status = "error"
            self._error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "status": self._status,
            "started_at": self._started_at,
            "error": self._error,
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._safe_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        # Close the input stream to unblock the for-loop in run()
        if hasattr(self, "_input_stream") and self._input_stream is not None:  # type: ignore[has-type]
            self._input_stream.close()  # type: ignore[has-type]
        # Close all output topics
        for t in self._topics:
            if t is not _NONE:
                t.close()
        self._status = "stopped"
        if self in Node._registry:
            Node._registry.remove(self)
