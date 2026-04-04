from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import pytest
from pydantic import BaseModel

from src.core.channel import Receiver, Sender, Channel, UIReceiver, UISender
from src.core.component import (
    PrimitiveComponent,
    ThreadedComponent,
    CompositeComponent,
    Status,
)
from src.core.frames import TextFrame
from src.core.graph import Edge, Graph, GraphManager, Node


class InitModel(BaseModel):
    v: int


class InT(NamedTuple):
    inp: Receiver[TextFrame] | None
    ui_in: UIReceiver[TextFrame]


class OutT(NamedTuple):
    out: Sender[TextFrame]
    ui_out: UISender[TextFrame]


class DemoComp(PrimitiveComponent[InT, OutT]):
    description = "demo"
    _registerable = True

    def __init__(self, p: Path, model: InitModel, opt: int = 1) -> None:
        super().__init__()
        self.p = p
        self.model = model
        self.opt = opt


class DemoThread(
    ThreadedComponent[tuple[Receiver[TextFrame] | None], tuple[Sender[TextFrame]]]
):
    _registerable = True

    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def setup(self, outputs) -> None:
        if outputs and outputs[0] is not None:
            outputs[0].send(TextFrame.new(text="boot"))
        self.started = True

    def run(self, inputs, outputs) -> None:
        while not self.stop_event.is_set():
            time.sleep(0.001)
            break


def test_component_helpers_and_from_args() -> None:
    init_types = DemoComp.get_init_types()
    assert "p" in init_types and "model" in init_types
    comp = DemoComp(Path("demo"), InitModel(v=1))
    assert "inp" in comp.get_input_types()
    assert "out" in comp.get_output_types()
    assert "ui_in" in comp.get_ui_input_types()
    assert "ui_out" in comp.get_ui_output_types()
    assert DemoComp.get_options({}) == {}
    c = DemoComp.from_args({"p": "a/b", "model": {"v": 2}, "opt": 3})
    assert isinstance(c.p, Path)
    assert c.model.v == 2
    assert c.opt == 3
    with pytest.raises(ValueError):
        DemoComp.from_args({"p": "x"})
    assert DemoComp._resolve_tuple_types(tuple[int, str]) == {"1": int, "2": str}
    assert DemoComp._resolve_tuple_types(None) == {}


def test_threaded_component_lifecycle(monkeypatch) -> None:
    comp = DemoThread()
    ch = Channel[TextFrame]()
    snd = Sender(ch)
    comp.start((None,), (snd,))
    time.sleep(0.01)
    assert comp.status in {Status.RUNNING, Status.STOPPED, Status.SETUP}
    assert comp.get_ident() is not None
    comp.stop()
    comp.stop()


def test_composite_component_and_graph_manager(monkeypatch) -> None:
    classes = {"DemoThread": DemoThread, "DemoComp": DemoComp}
    monkeypatch.setattr(
        PrimitiveComponent, "registered_subclasses", classmethod(lambda cls: classes)
    )

    n1 = Node(id_="n1", type="DemoThread", init_args={})
    n2 = Node(id_="n2", type="DemoThread", init_args={})
    g = Graph(
        nodes={"n1": n1, "n2": n2},
        edges=[
            Edge(source_node="n1", source_slot="1", target_node="n2", target_slot="1")
        ],
    )
    gm = GraphManager(g)
    assert gm.get_node("n1") is not None
    assert gm.get_node("x") is None
    assert gm.update_node("x", 1, 2) is None
    assert gm.update_node("n1", 1, 2) is not None
    assert gm.get_node_output("n1")
    assert gm.get_node_input("n2")
    gm.run()
    assert gm.ui_senders() == {}
    assert gm.ui_receivers() == {}
    gm.stop()

    gm.add_edge(
        Edge(source_node="n1", source_slot="1", target_node="n2", target_slot="1")
    )
    gm.delete_edge(gm.graph.edges[-1])
    grouped = GraphManager._group([(("a", "o"), ("b", "i"))])
    assert len(grouped) == 1
    assert gm.components()
    assert gm.sender_handles() is not None
    assert gm.receiver_handles() is not None

    sg = Graph(nodes={"n1": n1}, edges=[])
    cid, _ = gm.add_composite_node("Comp", sg)
    assert gm.get_node(cid) is not None
    gm.delete_node(cid)


def test_composite_component_start_stop(monkeypatch) -> None:
    classes = {"DemoThread": DemoThread}
    monkeypatch.setattr(
        PrimitiveComponent, "registered_subclasses", classmethod(lambda cls: classes)
    )
    n1 = Node(id_="a", type="DemoThread", init_args={})
    sub = Graph(nodes={"a": n1}, edges=[])
    comp = CompositeComponent("C", sub)
    assert "1" in comp.get_input_types()
    assert "1" in comp.get_output_types()
    comp.start((), ())
    assert comp.status == Status.RUNNING
    comp.stop()
    assert comp.status == Status.STOPPED
