import threading
from typing import Protocol

from .stream import Stream


class Streamable[T](Protocol):
    def stream(self) -> Stream[T]:
        raise NotImplementedError


class Base[T](Streamable[T]):
    def __init__(self):
        self._items: list[T] = []
        self._offset = 0
        self._condition = threading.Condition()
        self._cursors: dict[int, int] = {}
        self._next_sub_id = 0

    def run(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def stream(self) -> Stream[T]:
        return Stream(self)

    def send(self, item: T) -> None:
        with self._condition:
            self._items.append(item)
            self._condition.notify_all()

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

    def _gc(self):
        if not self._cursors:
            return
        drop = min(self._cursors.values()) - self._offset
        if drop > 0:
            del self._items[:drop]
            self._offset += drop
