"""Fast Channel — drop-in replacement for src.core.channel using a Rust ring buffer.

Same interface as Channel/Sender/Receiver/ReceiverIterator in src.core.channel.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Iterator

from fast_channel._fast_channel import RustChannel

if TYPE_CHECKING:
    from src.core.component import ThreadedComponent


class Channel[T]:
    def __init__(self, capacity: int = 65536) -> None:
        self._inner = RustChannel(capacity)
        # Keep _items-like interface for buffer_depth (used by Sender)
        self._rust = self._inner

    def _send(self, item: T) -> None:
        self._inner.send(item)

    def _register(self, sub_id: int, latest: bool = True) -> None:
        self._inner.register(sub_id, latest)

    def _wait_and_get(self, sub_id: int, stop_event: threading.Event) -> T | None:
        return self._inner.wait_and_get(sub_id, stop_event.is_set)

    def _try_get(self, sub_id: int) -> T | None:
        return self._inner.try_get(sub_id)

    def _fast_forward(self, sub_id: int) -> None:
        self._inner.fast_forward(sub_id)

    def _unregister(self, sub_id: int) -> None:
        self._inner.unregister(sub_id)

    def _gc(self) -> None:
        pass  # Ring buffer doesn't need GC


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
        self._last_send_time = time.time()

    @property
    def buffer_depth(self) -> int:
        return 0  # Ring buffer doesn't track this the same way


class ReceiverIterator[T]:
    """Eager-registering iterator matching src.core.channel.ReceiverIterator."""

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
        if self._sub_id is None:
            return 0
        return self._channel._inner.lag(self._sub_id)
