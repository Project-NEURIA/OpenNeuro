from __future__ import annotations

import threading

import pytest

from src.core.channel import Channel, Receiver, Sender
from src.core.frames import TextFrame


def test_channel_send_receive_and_gc(dummy_subscriber) -> None:
    channel = Channel[int]()
    sender = Sender(channel)
    receiver = Receiver(channel)

    sender.send(1)
    sender.send(2)
    iterator = receiver(dummy_subscriber, latest=False)
    assert next(iterator) == 1
    assert next(iterator) == 2
    assert receiver.lag == 0
    assert sender.buffer_depth == 0
    iterator.close()


def test_channel_non_blocking_and_fast_forward(dummy_subscriber) -> None:
    channel = Channel[int]()
    receiver = Receiver(channel)
    sender = Sender(channel)

    sender.send(10)
    sender.send(20)
    sender.send(30)

    iterator = receiver(dummy_subscriber, newest=True, no_block=True, latest=False)
    assert next(iterator) == 30
    assert next(iterator) is None
    iterator.close()


def test_channel_wait_stop_and_unregister_idempotent() -> None:
    channel = Channel[int]()
    stop_event = threading.Event()
    channel._register(1, latest=True)
    stop_event.set()
    assert channel._wait_and_get(1, stop_event) is None
    channel._unregister(1)
    channel._unregister(1)


def test_receiver_iterator_stop_event_finishes(dummy_subscriber) -> None:
    channel = Channel[str]()
    receiver = Receiver(channel)
    iterator = receiver(dummy_subscriber)
    assert iter(iterator) is iterator
    dummy_subscriber.stop_event.set()
    assert next(iterator) is None
    with pytest.raises(StopIteration):
        next(iterator)


def test_sender_connect_and_metrics() -> None:
    c1 = Channel[TextFrame]()
    c2 = Channel[TextFrame]()
    sender = Sender(c1)
    sender.connect(c2)
    frame = TextFrame.new(text="x")
    sender.send(frame)
    assert sender._msg_count == 1
    assert sender._byte_count > 0
    assert sender._last_send_time > 0
    assert sender.buffer_depth == 2


def test_receiver_lag_without_subscriber_or_cursor(dummy_subscriber) -> None:
    channel = Channel[int]()
    receiver = Receiver(channel)
    assert receiver.lag == 0
    receiver(dummy_subscriber)
    channel._unregister(id(dummy_subscriber))
    assert receiver.lag == 0
