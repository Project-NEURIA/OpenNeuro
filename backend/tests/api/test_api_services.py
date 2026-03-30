from __future__ import annotations

import json
import types

import pytest
from fastapi import HTTPException

from src.api import dep
from src.api.component import controller as component_controller
from src.api.component import service as component_service
from src.api.env import controller as env_controller
from src.api.graph.edge import service as edge_service
from src.api.graph.run import service as run_service
from src.api.graph.save import service as save_service
from src.api.logs import controller as logs_controller
from src.api.metrics.service import MetricsCollector
from src.api.project import service as project_service
from src.core.channel import Channel, Receiver, Sender
from src.core.component import PrimitiveComponent, Tag
from src.core.frames import TextFrame
from src.core.graph import Edge, Graph, Node
from src.main import _copy_presets


class FakeManager:
    def __init__(self) -> None:
        n1 = Node(id_="a", type="X", init_args={})
        n2 = Node(id_="b", type="Y", init_args={})
        self.graph = Graph(nodes={"a": n1, "b": n2}, edges=[])
        self._run_called = False
        self._stop_called = False
        self._reset_called = False
        channel = Channel()
        self._sender = Sender(channel)
        self._receiver = Receiver(channel)

    def get_node(self, node_id: str):
        return self.graph.nodes.get(node_id)

    def get_node_output(self, _node_id: str):
        return {"out": Sender}

    def get_node_input(self, _node_id: str):
        return {"in": Receiver}

    def add_edge(self, edge: Edge) -> None:
        self.graph.edges.append(edge)

    def delete_edge(self, edge: Edge) -> None:
        self.graph.edges.remove(edge)

    def run(self) -> None:
        self._run_called = True

    def stop(self) -> None:
        self._stop_called = True

    def reset(self, _graph: Graph) -> None:
        self._reset_called = True

    def components(self):
        return {
            "a": types.SimpleNamespace(
                type_="A", status=types.SimpleNamespace(value="running")
            )
        }

    def sender_handles(self):
        return {("a", "out"): self._sender}

    def receiver_handles(self):
        return {("a", "in"): self._receiver}


def test_dep_get_manager() -> None:
    request = types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(manager=123))
    )
    assert dep.get_manager(request) == 123


def test_component_controller_and_service(monkeypatch) -> None:
    class FakeComponentClass:
        tags = Tag(io={"source"}, functionality={"misc"})
        description = "desc"

        @classmethod
        def get_init_types(cls):
            return {"value": int}

        @classmethod
        def get_input_types(cls):
            return {"inp": str | int}

        @classmethod
        def get_output_types(cls):
            return {"out": list[str]}

        @classmethod
        def get_ui_input_types(cls):
            return {"ui_in": str}

        @classmethod
        def get_ui_output_types(cls):
            return {"ui_out": bytes}

        @classmethod
        def get_options(cls, values):
            return {"seen": values}

        @classmethod
        def _class_input_types(cls):
            return cls.get_input_types()

        @classmethod
        def _class_output_types(cls):
            return cls.get_output_types()

        @classmethod
        def _class_ui_input_types(cls):
            return cls.get_ui_input_types()

        @classmethod
        def _class_ui_output_types(cls):
            return cls.get_ui_output_types()

    original = component_controller.service.list_components
    monkeypatch.setattr(
        component_controller.service,
        "list_components",
        lambda: {"Fake": FakeComponentClass},
    )
    out = component_controller.list_components()
    assert out[0].type_ == "Fake"
    assert component_controller.is_type("int") is True
    assert component_controller.is_type("TypeThatDoesNotExist") is False
    assert component_controller.is_subtype("bool", "int") is True
    assert component_controller.is_subtype("Unknown", "int") is False
    assert component_controller._type_name(int | str).startswith("Union")
    assert component_controller.get_options("Fake", {"k": "v"}) == {"seen": {"k": "v"}}
    with pytest.raises(HTTPException):
        component_controller.get_options("Missing", {})

    monkeypatch.setattr(component_controller.service, "list_components", original)
    monkeypatch.setattr(
        PrimitiveComponent, "registered_subclasses", lambda: {"A": object}
    )
    assert component_service.list_components() == {"A": object}


def test_component_controller_project_branches(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(component_controller.service, "list_components", lambda: {})
    monkeypatch.setattr(component_controller, "PROJECTS_DIR", tmp_path)

    valid = tmp_path / "ValidProject"
    valid.mkdir()
    (valid / "graph.json").write_text('{"nodes": {}, "edges": []}', encoding="utf-8")
    (valid / "thumbnail.png").write_bytes(b"x")

    invalid = tmp_path / "BrokenProject"
    invalid.mkdir()
    (invalid / "graph.json").write_text("{not-json", encoding="utf-8")

    (tmp_path / "not_a_dir.txt").write_text("x", encoding="utf-8")

    out = component_controller.list_components()
    assert [item.type_ for item in out] == ["ValidProject"]
    assert out[0].is_composite is True
    assert out[0].has_thumbnail is True


def test_env_controller(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(env_controller, "ENV_PATH", env_path)
    assert env_controller.get_env().content == ""
    env_controller.put_env(env_controller.EnvBody(content="A=1"))
    assert env_controller.get_env().content == "A=1"


def test_edge_run_save_services(tmp_path, monkeypatch) -> None:
    manager = FakeManager()
    edge_service.create_edge(manager, "a", "out", "b", "in")
    assert len(manager.graph.edges) == 1
    assert len(edge_service.list_edges(manager)) == 1
    with pytest.raises(ValueError):
        edge_service.create_edge(manager, "a", "missing", "b", "in")
    with pytest.raises(ValueError):
        edge_service.create_edge(manager, "a", "out", "b", "missing")
    with pytest.raises(KeyError):
        edge_service.create_edge(manager, "missing", "out", "b", "in")
    with pytest.raises(KeyError):
        edge_service.create_edge(manager, "a", "out", "missing", "in")
    with pytest.raises(ValueError):
        edge_service.create_edge(manager, "a", "out", "b", "in")

    edge_service.delete_edge(manager, "a", "out", "b", "in")
    with pytest.raises(KeyError):
        edge_service.delete_edge(manager, "a", "out", "b", "in")

    run_service.start_all(manager)
    run_service.stop_all(manager)
    assert manager._run_called is True
    assert manager._stop_called is True

    monkeypatch.setattr(save_service, "PROJECTS_DIR", tmp_path)
    p = tmp_path / "demo"
    p.mkdir()
    save_service.save_graph("demo", Graph(nodes={}, edges=[]))
    assert (p / "graph.json").exists()


def test_logs_controller(monkeypatch) -> None:
    entry = types.SimpleNamespace(seq=1, timestamp=2.0, stream="stdout", text="x")
    fake_store = types.SimpleNamespace(
        get_entries=lambda node_id, after, limit: [entry]
    )
    monkeypatch.setattr(logs_controller, "get_log_store", lambda: fake_store)
    out = logs_controller.get_component_logs("node")
    assert out.node_id == "node"
    assert out.entries[0].text == "x"


def test_metrics_collector_collect() -> None:
    manager = FakeManager()
    frame = TextFrame.new(text="hello")
    stop_event = types.SimpleNamespace(is_set=lambda: False)
    manager._receiver.blocking = False
    manager._receiver._wire(stop_event)
    manager._sender.send(frame)
    next(manager._receiver)

    collector = MetricsCollector()
    first = collector.collect(manager)
    second = collector.collect(manager)
    assert "a" in first.nodes
    assert second.nodes["a"].senders["out"].msg_count_delta == 0
    manager._receiver._unwire()


def test_project_service(monkeypatch, tmp_path) -> None:
    projects = tmp_path / "projects"
    config = tmp_path / "config.json"
    monkeypatch.setattr(project_service, "PROJECTS_DIR", projects)
    monkeypatch.setattr(project_service, "CONFIG_PATH", config)

    assert project_service.list_projects() == []
    project_service.create_project("demo")
    listed = project_service.list_projects()
    assert listed[0].name == "demo"
    assert listed[0].has_thumbnail is True

    manager = FakeManager()
    graph_path = projects / "demo" / "graph.json"
    graph_path.write_text(json.dumps({"nodes": {}, "edges": []}))
    project_service.start_project("demo", manager)
    assert manager._reset_called is True
    assert "demo" in config.read_text()

    project_service.close_project(manager)
    assert "null" in config.read_text()

    with pytest.raises(ValueError):
        project_service.start_project("missing", manager)

    project_service.delete_project("demo")
    assert not (projects / "demo").exists()


def test_copy_presets(monkeypatch, tmp_path) -> None:
    presets = tmp_path / "presets"
    projects = tmp_path / "projects"
    preset_a = presets / "A"
    preset_a.mkdir(parents=True)
    (preset_a / "graph.json").write_text("{}")
    (projects / "A").mkdir(parents=True)

    monkeypatch.setattr("src.main.PRESETS_DIR", presets)
    monkeypatch.setattr("src.main.PROJECTS_DIR", projects)
    _copy_presets()
    assert (projects / "A").exists()

    preset_b = presets / "B"
    preset_b.mkdir(parents=True)
    (preset_b / "graph.json").write_text("{}")
    (presets / "not_a_dir.txt").write_text("x")
    _copy_presets()
    assert (projects / "B").exists()

    monkeypatch.setattr("src.main.PRESETS_DIR", tmp_path / "none")
    _copy_presets()
