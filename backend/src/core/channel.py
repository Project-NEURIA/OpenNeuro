from __future__ import annotations

import sys
import threading
import time
from typing import Iterator

from src.core.frames import TextFrame


class Channel[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
        self._offset = 0
        self._condition = threading.Condition()
        self._cursors: dict[int, int] = {}
        self._newest_subs: dict[int, int] = {}

    def _send(self, item: T) -> None:
        with self._condition:
            if not self._cursors and not self._newest_subs:
                return
            self._items.append(item)
            # Newest-only channel: aggressively trim to 1 item
            if not self._cursors and len(self._items) > 1:
                self._offset += len(self._items) - 1
                self._items[:] = self._items[-1:]
            self._condition.notify_all()

    def _register(self, sub_id: int, newest: bool = False) -> None:
        with self._condition:
            head = self._offset + len(self._items)
            if newest:
                self._newest_subs[sub_id] = head
            else:
                self._cursors[sub_id] = head

    def _unregister(self, sub_id: int) -> None:
        """Idempotent."""
        with self._condition:
            removed = self._cursors.pop(sub_id, None)
            if removed is not None:
                self._gc()
            else:
                self._newest_subs.pop(sub_id, None)

    def _reregister(self, sub_id: int, newest: bool) -> None:
        """Atomically switch a subscriber between cursor and newest mode."""
        with self._condition:
            removed = self._cursors.pop(sub_id, None)
            if removed is None:
                self._newest_subs.pop(sub_id, None)
            else:
                self._gc()
            head = self._offset + len(self._items)
            if newest:
                self._newest_subs[sub_id] = head
            else:
                self._cursors[sub_id] = head

    def _get(
        self,
        sub_id: int,
        stop_event: threading.Event,
        blocking: bool = True,
    ) -> T | None:
        """Unified read. Dispatches based on subscriber type."""
        with self._condition:
            is_newest = sub_id in self._newest_subs
        if is_newest:
            return self._get_newest(sub_id, stop_event, blocking)
        return self._get_cursor(sub_id, stop_event, blocking)

    def _get_cursor(
        self,
        sub_id: int,
        stop_event: threading.Event,
        blocking: bool,
    ) -> T | None:
        with self._condition:
            index = self._cursors[sub_id]
            if blocking:
                while index >= self._offset + len(self._items):
                    self._condition.wait(0.1)
                    if stop_event.is_set():
                        return None
            else:
                if index >= self._offset + len(self._items):
                    return None
            item = self._items[index - self._offset]
            self._cursors[sub_id] = index + 1
            self._gc()
        return item

    def _get_newest(
        self,
        sub_id: int,
        stop_event: threading.Event,
        blocking: bool,
    ) -> T | None:
        with self._condition:
            last_seen = self._newest_subs[sub_id]
            head = self._offset + len(self._items) - 1
            if blocking:
                while head < last_seen or not self._items:
                    self._condition.wait(0.1)
                    if stop_event.is_set():
                        return None
                    head = self._offset + len(self._items) - 1
            else:
                if head < last_seen or not self._items:
                    return None
            item = self._items[-1]
            self._newest_subs[sub_id] = head + 1
        return item

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
        self._stopped: bool = False

    def send(self, item: T) -> None:
        if self._stopped:
            return
        for ch in self._channels:
            ch._send(item)
        self._msg_count += 1
        self._byte_count += sys.getsizeof(item)
        self._last_send_time = time.time()

    @property
    def buffer_depth(self) -> int:
        return sum(len(ch._items) for ch in self._channels)


class Receiver[T]:
    """Handle for receiving from a channel. Is itself the iterator."""

    def __init__(self, channel: Channel[T]) -> None:
        self._channel = channel
        self._msg_count: int = 0
        self._byte_count: int = 0
        self._sub_id: int | None = None
        self._stop_event: threading.Event | None = None
        self._wired: bool = False
        self._newest: bool = False
        self.blocking: bool = True

    @property
    def newest(self) -> bool:
        return self._newest

    @newest.setter
    def newest(self, value: bool) -> None:
        if self._newest == value:
            return
        self._newest = value
        if self._wired:
            self._channel._reregister(self._sub_id, newest=value)  # type: ignore[arg-type]

    def _wire(self, stop_event: threading.Event) -> None:
        """Register with the channel. Called by GraphManager.run()."""
        if self._wired:
            return
        self._sub_id = id(self)
        self._stop_event = stop_event
        self._channel._register(self._sub_id, newest=self._newest)
        self._wired = True

    def _unwire(self) -> None:
        """Unregister from the channel. Idempotent."""
        if not self._wired:
            return
        self._channel._unregister(self._sub_id)  # type: ignore[arg-type]
        self._wired = False

    def __iter__(self) -> Iterator[T | None]:
        return self

    def __next__(self) -> T | None:
        item = self._channel._get(
            self._sub_id,  # type: ignore[arg-type]
            self._stop_event,  # type: ignore[arg-type]
            blocking=self.blocking,
        )
        if item is not None:
            self._msg_count += 1
            self._byte_count += sys.getsizeof(item)
        return item

    def __del__(self) -> None:
        try:
            self._unwire()
        except Exception:
            pass

    @property
    def lag(self) -> int:
        sub_id = self._sub_id
        if sub_id is None or self._newest:
            return 0
        ch = self._channel
        with ch._condition:
            cursor = ch._cursors.get(sub_id)
            if cursor is None:
                return 0
            head = ch._offset + len(ch._items)
            return head - cursor


# -- UI channel markers --


class UISender[T](Sender[T]):
    """Marker base: data flows from component to the frontend node UI."""


class UIReceiver[T](Receiver[T]):
    """Marker base: data flows from the frontend node UI to the component."""


class UITextSender(UISender[TextFrame]):
    """Component sends text for display in the node UI."""


class UIVideoSender(UISender[bytes]):
    """Component sends JPEG bytes for display in the node UI."""


class UITextReceiver(UIReceiver[TextFrame]):
    """Component receives text typed by the user in the node UI."""


class UIKeystrokeReceiver(UIReceiver[TextFrame]):
    """Component receives individual keystrokes from the node UI."""
