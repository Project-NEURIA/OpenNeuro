from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from src.core.component import Component


class Channel[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
        self._offset = 0
        self._condition = threading.Condition()
        self._cursors: dict[int, int] = {}

    def _send(self, item: T) -> None:
        with self._condition:
            if not self._cursors:
                return
            self._items.append(item)
            self._condition.notify_all()

    def _register(self, sub_id: int) -> None:
        with self._condition:
            self._cursors[sub_id] = self._offset + len(self._items)

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


class Sender[T]:
    """Handle for sending to one or more channels."""

    def __init__(self, *channels: Channel[T]) -> None:
        self._channels: list[Channel[T]] = list(channels)
        self._msg_count: int = 0
        self._byte_count: int = 0
        self._last_send_time: float = 0.0

    def connect(self, channel: Channel[T]) -> None:
        self._channels.append(channel)

    def send(self, item: T) -> None:
        for ch in self._channels:
            ch._send(item)
        self._msg_count += 1
        self._byte_count += sys.getsizeof(item)
        self._last_send_time = time.monotonic()

    @property
    def buffer_depth(self) -> int:
        return sum(len(ch._items) for ch in self._channels)


class Receiver[T]:
    """Handle for receiving from a channel."""

    def __init__(self, channel: Channel[T]) -> None:
        self._channel = channel
        self._msg_count: int = 0
        self._byte_count: int = 0
        self._sub_id: int | None = None

    def __call__(
        self, subscriber: Component[Any, Any]
    ) -> Generator[T | None, None, None]:
        stop_event = subscriber.stop_event
        sub_id = id(subscriber)
        self._sub_id = sub_id
        self._channel._register(sub_id)
        try:
            while not stop_event.is_set():
                item = self._channel._wait_and_get(sub_id, stop_event)
                yield item
                if item is not None:
                    self._msg_count += 1
                    self._byte_count += sys.getsizeof(item)
            yield None
        finally:
            self._channel._unregister(sub_id)

    @property
    def lag(self) -> int:
        sub_id = self._sub_id
        if sub_id is None:
            return 0
        ch = self._channel
        with ch._condition:
            cursor = ch._cursors.get(sub_id)
            if cursor is None:
                return 0
            head = ch._offset + len(ch._items)
            return head - cursor
