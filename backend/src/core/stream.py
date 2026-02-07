from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Base


class Stream[T](Generator[T, None, None]):
    def __init__(self, parent: Base[T]) -> None:
        self._parent = parent
        self._sub_id, self._index = parent._register()
        self._closed = False

    def send(self, value: None) -> T:
        if self._closed:
            raise StopIteration
        item = self._parent._wait_and_get(self._index)
        self._index += 1
        self._parent._advance(self._sub_id, self._index)
        return item

    def throw(self, typ=None, val=None, tb=None):
        self.close()
        raise StopIteration

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._parent._unregister(self._sub_id)
