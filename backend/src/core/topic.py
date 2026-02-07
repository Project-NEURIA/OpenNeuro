from __future__ import annotations

import sys
import threading
import time
from collections.abc import Generator


class Topic[T]:
    _registry: list[Topic] = []

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
        self._closed: bool = False
        Topic._registry.append(self)

    def send(self, item: T) -> None:
        with self._condition:
            if self._closed:
                return
            self._items.append(item)
            self._msg_count += 1
            self._byte_count += sys.getsizeof(item)
            self._last_send_time = time.time()
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self in Topic._registry:
            Topic._registry.remove(self)

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "msg_count": self._msg_count,
            "byte_count": self._byte_count,
            "last_send_time": self._last_send_time,
            "buffer_depth": len(self._items),
            "subscribers": len(self._cursors),
        }

    @classmethod
    def all_topics(cls) -> list[Topic]:
        return list(cls._registry)

    def stream(self) -> Stream[T]:
        return Stream(self)

    def _register(self) -> tuple[int, int]:
        with self._condition:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            index = self._offset + len(self._items)
            self._cursors[sub_id] = index
            return sub_id, index

    def _wait_and_get(self, index: int) -> T:
        with self._condition:
            while index >= self._offset + len(self._items):
                if self._closed:
                    raise StopIteration
                self._condition.wait()
            if self._closed:
                raise StopIteration
        return self._items[index - self._offset]

    def _advance(self, sub_id: int, index: int) -> None:
        with self._condition:
            self._cursors[sub_id] = index
            self._gc()

    def _unregister(self, sub_id: int) -> None:
        with self._condition:
            self._cursors.pop(sub_id, None)
            self._gc()

    def _gc(self) -> None:
        if not self._cursors:
            return
        drop = min(self._cursors.values()) - self._offset
        if drop > 0:
            del self._items[:drop]
            self._offset += drop


class Stream[T](Generator[T, None, None]):
    def __init__(self, channel: Topic[T]) -> None:
        self._channel = channel
        self._sub_id, self._index = channel._register()
        self._closed = False

    def send(self, value: None) -> T:
        if self._closed:
            raise StopIteration
        try:
            item = self._channel._wait_and_get(self._index)
        except StopIteration:
            self.close()
            raise
        self._index += 1
        self._channel._advance(self._sub_id, self._index)
        return item

    def throw(self, typ=None, val=None, tb=None):
        self.close()
        raise StopIteration

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._channel._unregister(self._sub_id)
        # Wake any blocked readers on the underlying topic
        with self._channel._condition:
            self._channel._condition.notify_all()
