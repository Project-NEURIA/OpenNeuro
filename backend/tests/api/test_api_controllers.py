from __future__ import annotations

import asyncio
import types

import pytest
from fastapi import HTTPException

from src.api.graph.edge import controller as edge_controller
from src.api.graph.node import controller as node_controller
from src.api.graph.run import controller as run_controller
from src.api.graph.save import controller as save_controller
from src.api.metrics import controller as metrics_controller
from src.api.project import controller as project_controller
from src.api.graph.node.dto import (
    NodeCreateRequest,
    NodeInitArgsUpdateRequest,
    NodeUpdateRequest,
    SubgraphCreateRequest,
)
from src.api.graph.edge.dto import EdgeCreateRequest
from src.core.graph import Graph, Node
from src.core.component import CompositeComponent
from src.core.channel import Channel, Receiver, Sender
from src.core.frames import TextFrame


class _Manager:
    def __init__(self) -> None:
        node = Node(id_="n1", type="A", init_args={}, x=1, y=2)
        self.graph = Graph(nodes={"n1": node}, edges=[])
        self._sender = Sender(Channel())
        self._receiver = Receiver(Channel())

    def component(self, _node_id: str):
        return types.SimpleNamespace(status=types.SimpleNamespace(value="running"))

    def components(self):
        return {
            "n1": types.SimpleNamespace(type_="A", status=types.SimpleNamespace(value="running"))
        }

    def sender_handles(self):
        return {("n1", "out"): self._sender}

    def receiver_handles(self):
        return {("n1", "inp"): self._receiver}


def test_edge_controller_list_create_delete(monkeypatch) -> None:
    manager = _Manager()
    monkeypatch.setattr(edge_controller.service, "list_edges", lambda m: [])
    assert edge_controller.list_edges(manager) == []

    req = EdgeCreateRequest(
        source_node="a", source_slot="out", target_node="b", target_slot="in"
    )
    monkeypatch.setattr(edge_controller.service, "create_edge", lambda *a, **k: None)
    out = edge_controller.create_edge(req, manager)
    assert out.source_node == "a"

    monkeypatch.setattr(
        edge_controller.service, "create_edge", lambda *a, **k: (_ for _ in ()).throw(KeyError("x"))
    )
    with pytest.raises(HTTPException):
        edge_controller.create_edge(req, manager)
    monkeypatch.setattr(
        edge_controller.service, "create_edge", lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
    )
    with pytest.raises(HTTPException):
        edge_controller.create_edge(req, manager)

    monkeypatch.setattr(edge_controller.service, "delete_edge", lambda *a, **k: None)
    assert edge_controller.delete_edge(req, manager) is None
    monkeypatch.setattr(
        edge_controller.service, "delete_edge", lambda *a, **k: (_ for _ in ()).throw(KeyError("x"))
    )
    with pytest.raises(HTTPException):
        edge_controller.delete_edge(req, manager)


def test_node_controller_paths(monkeypatch) -> None:
    manager = _Manager()
    node = manager.graph.nodes["n1"]

    monkeypatch.setattr(node_controller.service, "list_nodes", lambda m: {"n1": node})
    assert len(node_controller.list_nodes(manager)) == 1

    monkeypatch.setattr(node_controller.service, "get_node", lambda m, nid: None)
    with pytest.raises(HTTPException):
        node_controller.get_node("nope", manager)
    monkeypatch.setattr(node_controller.service, "get_node", lambda m, nid: node)
    assert node_controller.get_node("n1", manager).id == "n1"

    monkeypatch.setattr(
        node_controller.service,
        "create_node",
        lambda m, t, i: ("n1", node),
    )
    out = node_controller.create_node(NodeCreateRequest(type="A", init_args={}), manager)
    assert out.id == "n1"
    monkeypatch.setattr(
        node_controller.service, "create_node", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(HTTPException):
        node_controller.create_node(NodeCreateRequest(type="A", init_args={}), manager)

    monkeypatch.setattr(node_controller.service, "update_node", lambda *a, **k: None)
    with pytest.raises(HTTPException):
        node_controller.update_node("n1", NodeUpdateRequest(x=1, y=2), manager)
    monkeypatch.setattr(node_controller.service, "update_node", lambda *a, **k: node)
    assert node_controller.update_node("n1", NodeUpdateRequest(x=1, y=2), manager).id == "n1"

    monkeypatch.setattr(node_controller.service, "update_node_init_args", lambda *a, **k: None)
    with pytest.raises(HTTPException):
        node_controller.update_node_init_args("n1", NodeInitArgsUpdateRequest(init_args={}), manager)
    monkeypatch.setattr(node_controller.service, "update_node_init_args", lambda *a, **k: node)
    assert node_controller.update_node_init_args("n1", NodeInitArgsUpdateRequest(init_args={}), manager).id == "n1"

    called = {}
    monkeypatch.setattr(node_controller.service, "delete_node", lambda m, nid: called.setdefault("deleted", nid))
    node_controller.delete_node("n1", manager)
    assert called["deleted"] == "n1"

    monkeypatch.setattr(node_controller.service, "create_subgraph", lambda *a, **k: ("n1", node))
    assert node_controller.create_subgraph(SubgraphCreateRequest(node_ids=["n1"], name="S"), manager).id == "n1"
    monkeypatch.setattr(
        node_controller.service, "create_subgraph", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(HTTPException):
        node_controller.create_subgraph(SubgraphCreateRequest(node_ids=["n1"], name="S"), manager)

    monkeypatch.setattr(node_controller.service, "ungroup", lambda *a, **k: None)
    monkeypatch.setattr(node_controller.service, "list_nodes", lambda m: {"n1": node})
    assert len(node_controller.ungroup_node("n1", manager)) == 1
    monkeypatch.setattr(
        node_controller.service, "ungroup", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(HTTPException):
        node_controller.ungroup_node("n1", manager)


def test_node_controller_composite_response(monkeypatch) -> None:
    node = Node(id_="n1", type="A", init_args={}, sub_graph=Graph(nodes={}, edges=[]))
    comp = CompositeComponent("C", Graph(nodes={}, edges=[]))
    manager = types.SimpleNamespace(component=lambda _nid: comp)
    out = node_controller._node_response("n1", node, manager)
    assert out.is_composite is True


def test_run_save_metrics_project_controllers(monkeypatch, tmp_path) -> None:
    manager = _Manager()

    called = {}
    monkeypatch.setattr(run_controller.service, "start_all", lambda m: called.setdefault("start", True))
    monkeypatch.setattr(run_controller.service, "stop_all", lambda m: called.setdefault("stop", True))
    run_controller.start_all(manager)
    run_controller.stop_all(manager)
    assert called == {"start": True, "stop": True}

    monkeypatch.setattr(save_controller.AppConfig, "load_config", classmethod(lambda cls: types.SimpleNamespace(current_project=None)))
    assert save_controller.save_graph(manager) is None
    monkeypatch.setattr(save_controller.AppConfig, "load_config", classmethod(lambda cls: types.SimpleNamespace(current_project="demo")))
    monkeypatch.setattr(save_controller.service, "save_graph", lambda p, g: called.setdefault("save", p))
    save_controller.save_graph(manager)
    assert called["save"] == "demo"

    async def run_metrics():
        esr = await metrics_controller.get_metrics(manager)
        assert esr is not None
        agen = metrics_controller._stream(manager)
        first = await anext(agen)
        assert isinstance(first, str)
        second = await anext(agen)
        assert isinstance(second, str)
        await agen.aclose()

    asyncio.run(run_metrics())

    monkeypatch.setattr(project_controller.service, "list_projects", lambda: [])
    assert project_controller.list_projects() == []
    monkeypatch.setattr(project_controller.service, "create_project", lambda n: called.setdefault("create", n))
    project_controller.create_project(project_controller.ProjectCreateRequest(name="demo"))
    monkeypatch.setattr(project_controller.service, "delete_project", lambda n: called.setdefault("delete", n))
    project_controller.delete_project("demo")
    monkeypatch.setattr(project_controller.service, "start_project", lambda n, m: called.setdefault("start_project", n))
    assert project_controller.start_project("demo", manager).current_project == "demo"
    monkeypatch.setattr(project_controller.service, "close_project", lambda m: called.setdefault("close_project", True))
    project_controller.close_project(manager)
    monkeypatch.setattr(project_controller.AppConfig, "load_config", classmethod(lambda cls: types.SimpleNamespace(current_project="demo")))
    assert project_controller.get_current_project().current_project == "demo"

    monkeypatch.setattr(project_controller, "PROJECTS_DIR", tmp_path)
    (tmp_path / "demo").mkdir()
    thumb = tmp_path / "demo" / "thumbnail.png"
    thumb.write_bytes(b"x")
    resp = project_controller.get_thumbnail("demo")
    assert str(resp.path).endswith("thumbnail.png")
    with pytest.raises(HTTPException):
        project_controller.get_thumbnail("missing")
