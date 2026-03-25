from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Iterator

from src.core.frames import TextFrame

if TYPE_CHECKING:
    from src.core.component import ThreadedComponent


class Channel[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
        self._offset = 0
        self._condition = threading.Condition()
        self._cursors: dict[int, int] = {}

    def _send(self, item: T) -> None:
        with self._condition:
            self._items.append(item)
            self._condition.notify_all()

    def _register(self, sub_id: int, latest: bool = True) -> None:
        with self._condition:
            self._cursors[sub_id] = (
                self._offset + len(self._items) if latest else self._offset
            )

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

    def _try_get(self, sub_id: int) -> T | None:
        with self._condition:
            index = self._cursors[sub_id]
            if index >= self._offset + len(self._items):
                return None
            item = self._items[index - self._offset]
            self._cursors[sub_id] = index + 1
            self._gc()
        return item

    def _fast_forward(self, sub_id: int) -> None:
        with self._condition:
            head = self._offset + len(self._items)
            if head > self._cursors[sub_id] + 1:
                self._cursors[sub_id] = head - 1
                self._gc()

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
        # Use Unix epoch seconds to align with metrics.timestamp (time.time()).
        self._last_send_time = time.time()

    @property
    def buffer_depth(self) -> int:
        return sum(len(ch._items) for ch in self._channels)


class ReceiverIterator[T]:
    """Eager-registering iterator returned by ``Receiver.__call__``.

    Unlike a generator, the channel cursor is registered immediately on
    construction so that multiple iterators created in sequence all start
    at the same channel head — eliminating off-by-one frame lag when a
    component consumes several inputs from the same upstream.
    """

    def __init__(
            self,
            receiver: Receiver[T],
            channel: Channel[T],
            stop_event: threading.Event,
            sub_id: int,
            newest: bool,
            no_block: bool,
            latest: bool,
    ) -> None:
        self._receiver = receiver
        self._channel = channel
        self._stop_event = stop_event
        self._sub_id = sub_id
        self._newest = newest
        self._no_block = no_block
        self._done = False
        # Register cursor EAGERLY — this is the whole point.
        self._channel._register(sub_id, latest=latest)

    def __iter__(self) -> Iterator[T | None]:
        return self

    def __next__(self) -> T | None:
        if self._done:
            raise StopIteration
        if self._stop_event.is_set():
            self._finish()
            return None
        if self._newest:
            self._channel._fast_forward(self._sub_id)
        if self._no_block:
            item = self._channel._try_get(self._sub_id)
        else:
            item = self._channel._wait_and_get(self._sub_id, self._stop_event)
        if item is not None:
            self._receiver._msg_count += 1
            self._receiver._byte_count += sys.getsizeof(item)
        return item

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._channel._unregister(self._sub_id)

    def close(self) -> None:
        self._finish()

    def __del__(self) -> None:
        self._finish()


class Receiver[T]:
    """Handle for receiving from a channel."""

    def __init__(self, channel: Channel[T]) -> None:
        self._channel = channel
        self._msg_count: int = 0
        self._byte_count: int = 0
        self._sub_id: int | None = None

    def __call__(
            self,
            subscriber: ThreadedComponent[Any, Any],
            newest: bool = False,
            no_block: bool = False,
            latest: bool = True,
    ) -> Iterator[T | None]:
        """Return an iterator over items from the channel.

        The channel cursor is registered immediately (not deferred to the
        first ``next()`` call), so creating multiple iterators in sequence
        guarantees they all start at the same channel position.

        newest: skip to the latest item, dropping everything in between.
        no_block: return None immediately if nothing is available.
        latest: if True (default), start from the head. If False, start
                from the oldest available item (useful for reading emitted values).

        Yields None when the component is stopping or when no_block=True
        and there are no more new frames.
        """
        sub_id = id(subscriber)
        self._sub_id = sub_id
        return ReceiverIterator(
            receiver=self,
            channel=self._channel,
            stop_event=subscriber.stop_event,
            sub_id=sub_id,
            newest=newest,
            no_block=no_block,
            latest=latest,
        )

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
