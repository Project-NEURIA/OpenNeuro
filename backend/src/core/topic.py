from __future__ import annotations

import threading
from collections.abc import Generator


class Topic[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
        self._offset = 0
        self._condition = threading.Condition()
        self._cursors: dict[int, int] = {}
        self._next_sub_id = 0

    def send(self, item: T) -> None:
        with self._condition:
            self._items.append(item)
            self._condition.notify_all()

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
            if index >= self._offset + len(self._items):
                self._condition.wait()
        return self._items[index - self._offset]

    def _advance(self, sub_id: int, index: int) -> None:
        with self._condition:
            self._cursors[sub_id] = index
            self._gc()

    def _unregister(self, sub_id: int) -> None:
        with self._condition:
            del self._cursors[sub_id]
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
        item = self._channel._wait_and_get(self._index)
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
