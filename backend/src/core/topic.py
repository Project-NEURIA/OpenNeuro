from __future__ import annotations

import sys
import threading
import time
from typing import Generator

from pydantic import BaseModel



class TopicSnapshot(BaseModel):
    name: str
    msg_count: int
    byte_count: int
    last_send_time: float
    buffer_depth: int
    subscribers: int


class Topic[T]:

    def __init__(self, *, name: str | None = None) -> None:
        self._items: list[T] = []
        self._offset = 0
        self._condition = threading.Condition()
        self._cursors: dict[int, int] = {}
        self._next_sub_id = 0
        self.name: str = name or f"topic_{id(self):x}"
        self._msg_count: int = 0
        self._byte_count: int = 0
        self._last_send_time: float = 0.0

    def send(self, item: T) -> None:
        with self._condition:
            self._items.append(item)
            self._msg_count += 1
            self._byte_count += sys.getsizeof(item)
            self._last_send_time = time.time()
            self._condition.notify_all()

    def snapshot(self) -> TopicSnapshot:
        return TopicSnapshot(
            name=self.name,
            msg_count=self._msg_count,
            byte_count=self._byte_count,
            last_send_time=self._last_send_time,
            buffer_depth=len(self._items),
            subscribers=len(self._cursors),
        )

    def stream(self, stop_event: threading.Event) -> Generator[T | None, None, None]:
        """On GeneratorExit, stream is unregistered."""
        sub_id = self._register()
        try:
            while not stop_event.is_set():
                item = self._wait_and_get(sub_id, stop_event)
                yield item
            yield None
        finally:
            self._unregister(sub_id)

    def _register(self) -> int:
        with self._condition:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            self._cursors[sub_id] = self._offset + len(self._items)
            return sub_id

    def _wait_and_get(self, sub_id: int, stop_event: threading.Event) -> T | None:
        with self._condition:
            index = self._cursors[sub_id]
            while index >= self._offset + len(self._items):
                self._condition.wait(0.1)
                if stop_event.is_set():
                    return None
            item = self._items[index - self._offset]
            self._cursors[sub_id] = index + 1
            self._gc()
        return item

    def _unregister(self, sub_id: int) -> None:
        """Idempotent."""
        with self._condition:
            if sub_id not in self._cursors:
                return None
            self._cursors.pop(sub_id)
            self._gc()

    def _gc(self) -> None:
        if not self._cursors:
            return
        drop = min(self._cursors.values()) - self._offset
        if drop > 0:
            del self._items[:drop]
            self._offset += drop
