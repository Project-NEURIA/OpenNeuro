from __future__ import annotations

import threading

from src.core.channel import Channel, Receiver, Sender
from src.core.frames import TextFrame


def test_channel_send_receive_and_gc() -> None:
    channel = Channel[int]()
    sender = Sender(channel)
    receiver = Receiver(channel, threading.Event())

    sender.send(1)
    sender.send(2)
    assert next(receiver) == 1
    assert next(receiver) == 2
    assert receiver.lag == 0
    assert sender.buffer_depth == 0


def test_channel_non_blocking_and_fast_forward() -> None:
    channel = Channel[int]()
    receiver = Receiver(channel, threading.Event())
    sender = Sender(channel)
    receiver.newest = True
    receiver.blocking = False

    sender.send(10)
    sender.send(20)
    sender.send(30)

    assert next(receiver) == 30
    assert next(receiver) is None


def test_channel_wait_stop_and_unregister_idempotent() -> None:
    channel = Channel[int]()
    stop_event = threading.Event()
    channel._register(1, newest=True)
    stop_event.set()
    assert channel._get(1, stop_event) is None
    channel._unregister(1)
    channel._unregister(1)


def test_receiver_iterator_stop_event_finishes() -> None:
    channel = Channel[str]()
    stop_event = threading.Event()
    receiver = Receiver(channel, stop_event)
    assert iter(receiver) is receiver
    stop_event.set()
    assert next(receiver) is None


def test_sender_metrics_with_multiple_channels() -> None:
    c1 = Channel[TextFrame]()
    c2 = Channel[TextFrame]()
    _r1 = Receiver(c1, threading.Event())  # noqa: F841
    _r2 = Receiver(c2, threading.Event())  # noqa: F841
    sender = Sender(c1, c2)
    frame = TextFrame.new(text="x")
    sender.send(frame)
    assert sender._msg_count == 1
    assert sender._byte_count > 0
    assert sender._last_send_time > 0
    assert sender.buffer_depth == 2


def test_receiver_lag_without_subscriber_or_cursor() -> None:
    channel = Channel[int]()
    receiver = Receiver(channel, threading.Event())
    Sender(channel).send(5)
    assert receiver.lag == 1
