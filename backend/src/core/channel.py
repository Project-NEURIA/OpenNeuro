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
        self._newest: set[int] = set()

    def _send(self, item: T) -> None:
        with self._condition:
            if not self._cursors:
                return
            self._items.append(item)
            self._condition.notify_all()

    def _register(self, sub_id: int, newest: bool = False) -> None:
        with self._condition:
            head = self._offset + len(self._items)
            self._cursors[sub_id] = head
            if newest:
                self._newest.add(sub_id)

    def _unregister(self, sub_id: int) -> None:
        """Idempotent."""
        with self._condition:
            self._cursors.pop(sub_id, None)
            self._newest.discard(sub_id)
            self._gc()

    def _reregister(self, sub_id: int, newest: bool) -> None:
        """Switch a subscriber between cursor and newest mode."""
        with self._condition:
            if newest:
                self._newest.add(sub_id)
            else:
                self._newest.discard(sub_id)

    def _get(
        self,
        sub_id: int,
        stop_event: threading.Event,
        blocking: bool = True,
    ) -> T | None:
        with self._condition:
            # Fast-forward to latest if newest mode
            if sub_id in self._newest:
                head = self._offset + len(self._items)
                if head > 0 and self._cursors.get(sub_id, head) < head - 1:
                    self._cursors[sub_id] = head - 1
                    self._gc()

            index = self._cursors.get(sub_id)
            if index is None:
                return None

            if blocking:
                while index >= self._offset + len(self._items):
                    self._condition.wait(0.1)
                    if stop_event.is_set():
                        return None
                    index = self._cursors.get(sub_id, index)
            else:
                if index >= self._offset + len(self._items):
                    return None

            if index < self._offset:
                return None

            item = self._items[index - self._offset]
            self._cursors[sub_id] = index + 1
            self._gc()
        return item

    def _gc(self) -> None:
        if not self._cursors:
            return
        min_cursor = max(min(self._cursors.values()), self._offset)
        drop = min_cursor - self._offset
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


class UIVideoReceiver(UIReceiver[bytes]):
    """Component receives JPEG bytes from the frontend."""


class UIAudioSender(UISender[bytes]):
    """Component sends raw PCM bytes to the frontend."""


class UIAudioReceiver(UIReceiver[bytes]):
    """Component receives raw PCM bytes from the frontend."""


class UITextReceiver(UIReceiver[TextFrame]):
    """Component receives text typed by the user in the node UI."""


class UIKeystrokeReceiver(UIReceiver[TextFrame]):
    """Component receives individual keystrokes from the node UI."""
