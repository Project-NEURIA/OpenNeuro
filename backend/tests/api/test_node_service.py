from __future__ import annotations


import pytest

from src.api.graph.node import service as node_service
from src.api.graph.node.dto import NodeUpdateRequest
from src.core.component import PrimitiveComponent
from src.core.graph import Edge, Graph, Node


class _FakeComp:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeManager:
    def __init__(self) -> None:
        self._n1 = Node(id_="n1", type="A", init_args={}, x=1.0, y=2.0)
        self._n2 = Node(id_="n2", type="A", init_args={}, x=3.0, y=4.0)
        self.graph = Graph(
            nodes={"n1": self._n1, "n2": self._n2},
            edges=[
                Edge(
                    source_node="n1",
                    source_slot="out",
                    target_node="n2",
                    target_slot="inp",
                )
            ],
        )
        self._components = {"n1": _FakeComp(), "n2": _FakeComp()}
        self.reconciled = 0

    def add_primitive_node(self, type_, init_args):
        if type_ != "A":
            raise ValueError("bad")
        n = Node(id_="n3", type=type_, init_args=init_args)
        self.graph.nodes["n3"] = n
        self._components["n3"] = _FakeComp()
        return "n3", n

    def add_composite_node(self, type_, sub_graph, x=0.0, y=0.0, description=""):
        n = Node(id_="c1", type=type_, init_args={}, x=x, y=y, sub_graph=sub_graph)
        self.graph.nodes["c1"] = n
        self._components["c1"] = _FakeComp()
        return "c1", n

    def get_node(self, node_id):
        return self.graph.nodes.get(node_id)

    def update_node(self, node_id, x, y):
        n = self.graph.nodes.get(node_id)
        if n is None:
            return None
        n.x = x
        n.y = y
        return n

    def delete_node(self, node_id):
        self.graph.nodes.pop(node_id, None)
        self._components.pop(node_id, None)

    def update_primitive_node_init_args(self, node_id, init_args):
        n = self.graph.nodes.get(node_id)
        if n is None:
            return None, False
        n.init_args = init_args
        return n, False

    def components(self):
        return self._components

    def _reconcile(self):
        self.reconciled += 1


def test_node_service_basic_crud() -> None:
    m = _FakeManager()
    assert node_service.list_nodes(m)["n1"].id_ == "n1"
    assert node_service.get_node(m, "n1").id_ == "n1"
    assert node_service.get_node(m, "missing") is None
    assert node_service.create_node(m, "A", {})[0] == "n3"
    updated = node_service.update_node(m, "n1", NodeUpdateRequest(x=9, y=8))
    assert updated.x == 9
    node_service.delete_node(m, "n2")
    assert "n2" not in m.graph.nodes
    from src.api.ui.bridge import UIChannelBridge
    out = node_service.update_primitive_node_init_args(m, UIChannelBridge(), "n1", {"k": 1})
    assert out.init_args == {"k": 1}


def test_create_node_fallback_to_project(tmp_path, monkeypatch) -> None:
    m = _FakeManager()
    projects = tmp_path / "projects"
    monkeypatch.setattr(node_service, "PROJECTS_DIR", projects)
    p = projects / "Proj"
    p.mkdir(parents=True)
    g = Graph(nodes={}, edges=[])
    (p / "graph.json").write_text(g.model_dump_json())
    comp_id, comp_node = node_service.create_node(m, "Proj", {})
    assert comp_id == "c1"
    assert comp_node.sub_graph is not None
    with pytest.raises(ValueError):
        node_service.create_node(m, "UnknownProject", {})


def test_subgraph_create_and_ungroup(monkeypatch) -> None:
    m = _FakeManager()
    cid, cnode = node_service.create_subgraph(m, ["n1", "n2"], name="G")
    assert cid == "c1"
    assert cnode.sub_graph is not None
    assert "n1" not in m.graph.nodes
    assert m.reconciled >= 1

    class _Cls:
        @classmethod
        def from_args(cls, _args):
            return _FakeComp()

    monkeypatch.setattr(
        PrimitiveComponent, "registered_subclasses", lambda: {"A": _Cls}
    )
    node_service.ungroup(m, "c1")
    assert "c1" not in m.graph.nodes
    assert "n1" in m.graph.nodes

    # Exercise fallback rewiring branches when slot names do not contain dots.
    c2, _ = m.add_composite_node(
        "C2", Graph(nodes={"n1": m.graph.nodes["n1"]}, edges=[])
    )
    m.graph.edges.append(
        Edge(
            source_node="outside",
            source_slot="out",
            target_node=c2,
            target_slot="plain",
        )
    )
    m.graph.edges.append(
        Edge(
            source_node=c2, source_slot="plain", target_node="outside", target_slot="in"
        )
    )
    node_service.ungroup(m, c2)


def test_subgraph_rewires_boundary_edges(monkeypatch) -> None:
    m = _FakeManager()
    outside = Node(id_="outside", type="A", init_args={}, x=8.0, y=9.0)
    m.graph.nodes["outside"] = outside
    m._components["outside"] = _FakeComp()
    m.graph.edges.extend(
        [
            Edge(
                source_node="outside",
                source_slot="in",
                target_node="n1",
                target_slot="plain",
            ),
            Edge(
                source_node="n2",
                source_slot="out",
                target_node="outside",
                target_slot="sink",
            ),
            Edge(
                source_node="outside",
                source_slot="keep",
                target_node="outside",
                target_slot="stay",
            ),
        ]
    )

    cid, cnode = node_service.create_subgraph(m, ["n1", "n2"], name="G")
    assert cid == "c1"
    assert cnode.x == 2.0 and cnode.y == 3.0
    assert (
        Edge(
            source_node="outside",
            source_slot="in",
            target_node=cid,
            target_slot="n1.plain",
        )
        in m.graph.edges
    )
    assert (
        Edge(
            source_node=cid,
            source_slot="n2.out",
            target_node="outside",
            target_slot="sink",
        )
        in m.graph.edges
    )
    assert (
        Edge(
            source_node="outside",
            source_slot="keep",
            target_node="outside",
            target_slot="stay",
        )
        in m.graph.edges
    )

    class _Cls:
        @classmethod
        def from_args(cls, _args):
            return _FakeComp()

    monkeypatch.setattr(
        PrimitiveComponent, "registered_subclasses", lambda: {"A": _Cls}
    )
    node_service.ungroup(m, cid)
    assert (
        Edge(
            source_node="outside",
            source_slot="in",
            target_node="n1",
            target_slot="plain",
        )
        in m.graph.edges
    )
    assert (
        Edge(
            source_node="n2",
            source_slot="out",
            target_node="outside",
            target_slot="sink",
        )
        in m.graph.edges
    )


def test_subgraph_validation_errors() -> None:
    m = _FakeManager()
    with pytest.raises(ValueError):
        node_service.create_subgraph(m, ["missing"], name="Bad")
    with pytest.raises(ValueError):
        node_service.ungroup(m, "missing")
    with pytest.raises(ValueError):
        node_service.ungroup(m, "n1")
