from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Generator

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.core.component import Component


class SubscriberSnapshot(BaseModel):
    lag: int
    msg_count_delta: int
    byte_count_delta: int


class ChannelSnapshot(BaseModel):
    name: str
    msg_count_delta: int
    byte_count_delta: int
    last_send_time: float
    buffer_depth: int
    subscribers: dict[str, SubscriberSnapshot]


class Channel[T]:

    def __init__(self, *, name: str | None = None) -> None:
        self._items: list[T] = []
        self._offset = 0
        self._condition = threading.Condition()
        self._cursors: dict[int, int] = {}
        self._sub_msg_count_delta: dict[int, int] = {}
        self._sub_byte_count_delta: dict[int, int] = {}
        self.name: str = name or f"channel_{id(self):x}"
        self._msg_count_delta: int = 0
        self._byte_count_delta: int = 0
        self._last_send_time: float = 0.0

    def send(self, item: T) -> None:
        with self._condition:
            if not self._cursors: # stop data from accumulating when no one is listening
                return
            self._items.append(item)
            self._msg_count_delta += 1
            self._byte_count_delta += sys.getsizeof(item)
            self._last_send_time = time.time()
            self._condition.notify_all()

    def snapshot(self) -> ChannelSnapshot:
        with self._condition:
            total = self._offset + len(self._items)
            subs = {
                str(sub_id): SubscriberSnapshot(
                    lag=total - cursor,
                    msg_count_delta=self._sub_msg_count_delta.get(sub_id, 0),
                    byte_count_delta=self._sub_byte_count_delta.get(sub_id, 0),
                )
                for sub_id, cursor in self._cursors.items()
            }
            msg_count_delta = self._msg_count_delta
            byte_count_delta = self._byte_count_delta
            buffer_depth = len(self._items)
            self._msg_count_delta = 0
            self._byte_count_delta = 0
            for sub_id in self._sub_msg_count_delta:
                self._sub_msg_count_delta[sub_id] = 0
                self._sub_byte_count_delta[sub_id] = 0
        return ChannelSnapshot(
            name=self.name,
            msg_count_delta=msg_count_delta,
            byte_count_delta=byte_count_delta,
            last_send_time=self._last_send_time,
            buffer_depth=buffer_depth,
            subscribers=subs,
        )

    def stream(self, subscriber: Component[..., ...]) -> Generator[T | None, None, None]:
        """On GeneratorExit, stream is unregistered."""
        stop_event = subscriber.stop_event
        sub_id = self._register(subscriber)
        try:
            while not stop_event.is_set():
                item = self._wait_and_get(sub_id, stop_event)
                yield item
            yield None
        finally:
            self._unregister(sub_id)

    def _register(self, subscriber: Component[..., ...]) -> int:
        sub_id = id(subscriber)
        with self._condition:
            self._cursors[sub_id] = self._offset + len(self._items)
            self._sub_msg_count_delta[sub_id] = 0
            self._sub_byte_count_delta[sub_id] = 0
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
            self._sub_msg_count_delta[sub_id] = self._sub_msg_count_delta.get(sub_id, 0) + 1
            self._sub_byte_count_delta[sub_id] = self._sub_byte_count_delta.get(sub_id, 0) + sys.getsizeof(item)
            self._gc()
        return item

    def _unregister(self, sub_id: int) -> None:
        """Idempotent."""
        with self._condition:
            if sub_id not in self._cursors:
                return None
            self._cursors.pop(sub_id)
            self._sub_msg_count_delta.pop(sub_id, None)
            self._sub_byte_count_delta.pop(sub_id, None)
            self._gc()

    def _gc(self) -> None:
        if not self._cursors:
            return
        drop = min(self._cursors.values()) - self._offset
        if drop > 0:
            del self._items[:drop]
            self._offset += drop
