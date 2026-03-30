from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

from src.core.channel import Channel, Receiver, Sender
from src.core.component import (
    CompositeComponent,
    PrimitiveComponent,
    Status,
    ThreadedComponent,
)
from src.core.graph import Graph, GraphManager, Node


class _StubInputs(NamedTuple):
    data: Receiver[str]


class _StubOutputs(NamedTuple):
    data: Sender[str]


class _StubComponent(ThreadedComponent[_StubInputs, _StubOutputs]):
    def run(self, inputs: _StubInputs, outputs: _StubOutputs) -> None:
        return None


class _ProbeComponent(ThreadedComponent[tuple[()], tuple[()]]):
    def __init__(self, *, stop_during_setup: bool = False) -> None:
        super().__init__()
        self.stop_during_setup = stop_during_setup
        self.ran = False

    def setup(self) -> None:
        if self.stop_during_setup:
            self.stop_event.set()

    def run(self, inputs: tuple[()], outputs: tuple[()]) -> None:
        self.ran = True


def test_channel_remaining_paths(monkeypatch) -> None:
    channel = Channel[int]()
    sender = Sender(channel)

    sender.send(1)
    assert channel._items == []

    stop_event = threading.Event()
    cursor = Receiver(channel)
    cursor._wire(stop_event)
    cursor.blocking = False
    assert next(cursor) is None

    newest = Receiver(channel)
    newest.newest = True
    newest._wire(stop_event)
    newest.newest = False
    newest.newest = False
    assert newest._sub_id in channel._cursors

    channel._items = [10]
    channel._offset = 1
    channel._cursors[cursor._sub_id] = 0  # type: ignore[index]
    assert channel._get_cursor(cursor._sub_id, stop_event, blocking=False) is None  # type: ignore[arg-type]

    waiting_channel = Channel[int]()
    newest_blocking = Receiver(waiting_channel)
    newest_blocking.newest = True
    newest_blocking._wire(stop_event)
    monkeypatch.setattr(
        waiting_channel._condition, "wait", lambda timeout=None: stop_event.set()
    )
    assert next(newest_blocking) is None

    continue_channel = Channel[int]()
    continue_newest = Receiver(continue_channel)
    continue_newest.newest = True
    continue_newest._wire(threading.Event())
    wait_calls: list[float | None] = []

    def _wake_with_item(timeout: float | None = None) -> None:
        wait_calls.append(timeout)
        continue_channel._items.append(99)

    monkeypatch.setattr(continue_channel._condition, "wait", _wake_with_item)
    assert next(continue_newest) == 99
    assert wait_calls == [0.1]

    sender._stopped = True
    sender.send(2)
    assert sender._msg_count == 1

    bad_receiver = Receiver(channel)
    bad_receiver._wired = True
    bad_receiver._sub_id = 999
    bad_receiver._unwire = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
    bad_receiver.__del__()


def test_threaded_component_remaining_paths(monkeypatch) -> None:
    unregister_calls: list[str] = []
    monkeypatch.setattr(
        "src.core.component.get_log_store",
        lambda: SimpleNamespace(
            unregister_thread=lambda: unregister_calls.append("unregister")
        ),
    )

    stopped_in_setup = _ProbeComponent(stop_during_setup=True)
    stopped_in_setup._safe_run((), ())
    assert stopped_in_setup.ran is False
    assert stopped_in_setup.status == Status.STOPPED

    joins: list[float] = []
    started: list[bool] = []

    class _Thread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.ident = 123

        def start(self) -> None:
            started.append(True)

        def join(self, timeout=None) -> None:
            joins.append(timeout)

    comp = _ProbeComponent()
    comp._thread = SimpleNamespace(
        join=lambda timeout=None: joins.append(timeout), ident=7
    )
    monkeypatch.setattr("src.core.component.threading.Thread", _Thread)
    comp.start((), ())
    assert joins == [5.0]
    assert started == [True]
    assert unregister_calls == ["unregister"]


def test_composite_component_ui_types_and_graph_run_uses_dicts() -> None:
    registry = {"StubComponent": _StubComponent}
    sub_graph = Graph(
        nodes={"inner": Node(id_="inner", type="StubComponent", init_args={})},
        edges=[],
    )

    with patch.object(
        PrimitiveComponent, "registered_subclasses", return_value=registry
    ):
        composite = CompositeComponent("Composite", sub_graph)
        assert composite.get_ui_input_types() == {}
        assert composite.get_ui_output_types() == {}

        manager = GraphManager(Graph(nodes={}, edges=[]))
        node_id, _ = manager.add_composite_node("Composite", sub_graph)
        captured: dict[str, object] = {}
        component = manager.components()[node_id]
        component.start = lambda inputs, outputs: captured.update(  # type: ignore[method-assign]
            {"inputs": inputs, "outputs": outputs}
        )

        manager.run()

    assert isinstance(captured["inputs"], dict)
    assert isinstance(captured["outputs"], dict)
