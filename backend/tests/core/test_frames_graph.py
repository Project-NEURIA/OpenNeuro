from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from pydantic import BaseModel, model_validator

from src.core.channel import Channel, Receiver, Sender, UIReceiver, UISender
from src.core.component import (
    Component,
    CompositeComponent,
    PrimitiveComponent,
    Status,
    ThreadedComponent,
)
from src.core.frames import (
    AudioDataFormat,
    AudioFrame,
    BodyPoseFrame,
    BonePose,
    CameraParamsFrame,
    DepthFrame,
    GoalFrame,
    InterruptFrame,
    MessageFrame,
    ObjectDetectionFrame,
    ObjectLocationFrame,
    ObjectSegmentationFrame,
    RequestFrame,
    StereoCameraParamsFrame,
    StereoVideoFrame,
    TextFrame,
    ToolCall,
    ToolDef,
    ToolResult,
    VideoDataFormat,
    VideoFrame,
)
from src.core.graph import Edge, Graph, GraphManager, Node
from src.lib.misc.do_nothing import DoNothingInputs


class _ScalarModel(BaseModel):
    value: int

    @model_validator(mode="before")
    @classmethod
    def _coerce_scalar(cls, value: object) -> object:
        if isinstance(value, int):
            return {"value": value}
        return value


class _ModelComponent(PrimitiveComponent[tuple[()], tuple[()]]):
    def __init__(
        self,
        model: _ScalarModel | None = None,
        path: Path | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.path = path


class _ThreadBoom(ThreadedComponent[tuple[()], tuple[()]]):
    def run(self, inputs: tuple[()], outputs: tuple[()]) -> None:
        raise RuntimeError("boom")


class _CompositeIOInputs(NamedTuple):
    plain: Receiver[TextFrame] | None
    maybe: Receiver[TextFrame] | None
    ui_text: UIReceiver[TextFrame]


class _CompositeIOOutputs(NamedTuple):
    out: Sender[TextFrame]
    ui_text: UISender[TextFrame]


class _CompositeInner(ThreadedComponent[_CompositeIOInputs, _CompositeIOOutputs]):
    _registerable = True

    def __init__(self) -> None:
        super().__init__()
        self.started_inputs = None
        self.started_outputs = None
        self._ident_calls = 0

    def start(self, inputs: _CompositeIOInputs, outputs: _CompositeIOOutputs) -> None:
        self.started_inputs = inputs
        self.started_outputs = outputs
        self._status = Status.RUNNING

    def get_ident(self) -> int | None:
        self._ident_calls += 1
        return None if self._ident_calls == 1 else 321

    def run(self, inputs: _CompositeIOInputs, outputs: _CompositeIOOutputs) -> None:
        return None


class _OutputOnlyComp(PrimitiveComponent[tuple[()], tuple[Sender[TextFrame]]]):
    def __init__(self) -> None:
        super().__init__()


class _GenericPair[T](NamedTuple):
    value: T
    label: str


def test_component_generic_resolution_and_from_args() -> None:
    assert (
        Component._resolve_tuple_types(DoNothingInputs[int])["input"] == Receiver[int]
    )
    assert Component._resolve_tuple_types(_GenericPair[int])["label"] is str
    assert Component._resolve_tuple_types(tuple[()]) == {}
    assert _ModelComponent._get_type_param(99) is None

    empty = _ModelComponent.from_args({})
    assert empty.model is None and empty.path is None
    empty.start((), ())
    assert empty.status == Status.RUNNING

    from_dict = _ModelComponent.from_args({"model": {"value": 2}, "path": "x/y"})
    assert from_dict.model is not None and from_dict.model.value == 2
    assert from_dict.path == Path("x/y")
    assert from_dict.type_ == "_ModelComponent"

    existing = _ScalarModel(value=3)
    from_model = _ModelComponent.from_args({"model": existing})
    assert from_model.model is existing

    from_scalar = _ModelComponent.from_args({"model": 4})
    assert from_scalar.model is not None and from_scalar.model.value == 4


def test_threaded_component_exception_and_early_return(monkeypatch) -> None:
    unregistered: list[str] = []
    monkeypatch.setattr(
        "src.core.component.get_log_store",
        lambda: types.SimpleNamespace(
            unregister_thread=lambda: unregistered.append("x")
        ),
    )

    comp = _ThreadBoom()
    comp.start((), ())
    assert comp._thread is not None
    comp._thread.join(timeout=1)
    assert comp.status == Status.STOPPED
    assert unregistered == ["x"]

    comp2 = _ThreadBoom()
    comp2._status = Status.RUNNING
    comp2.start((), ())
    assert comp2._thread is None


def test_composite_component_additional_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        PrimitiveComponent,
        "registered_subclasses",
        classmethod(lambda cls: {"Known": _CompositeInner}),
    )

    sub_graph = Graph(
        nodes={
            "n1": Node(id_="n1", type="Known", init_args={}),
            "skip": Node(id_="skip", type="Missing", init_args={}),
        },
        edges=[
            Edge(
                source_node="n1",
                source_slot="out",
                target_node="n1",
                target_slot="plain",
            )
        ],
    )
    comp = CompositeComponent("Wrap", sub_graph)
    assert comp.type_ == "Wrap"
    assert comp.get_input_types()["n1.maybe"] == Receiver[TextFrame] | None
    assert "n1.out" not in comp.get_output_types()

    startable = CompositeComponent(
        "WrapStart",
        Graph(nodes={"n1": Node(id_="n1", type="Known", init_args={})}, edges=[]),
    )

    recv = Receiver(Channel())
    send = Sender(Channel())
    named_inputs = {"n1.plain": recv, "n1.maybe": recv}
    named_outputs = {"n1.out": send}
    startable.start(named_inputs, named_outputs)
    assert startable.status == Status.RUNNING
    assert startable._inner_manager is not None
    assert startable._inner_manager.receiver_handles()[("n1", "plain")] is recv
    assert startable._inner_manager.receiver_handles()[("n1", "maybe")] is recv
    assert startable._inner_manager.sender_handles()[("n1", "out")] is send
    startable.stop()

    comp2 = CompositeComponent(
        "Wrap2",
        Graph(nodes={"n1": Node(id_="n1", type="Known", init_args={})}, edges=[]),
    )
    comp2.start((recv, recv), (send,))
    assert comp2._inner_manager is not None
    assert ("n1", "plain") not in comp2._inner_manager.receiver_handles()
    assert ("n1", "maybe") not in comp2._inner_manager.receiver_handles()
    assert ("n1", "out") in comp2._inner_manager.sender_handles()
    assert comp2._inner_manager.sender_handles()[("n1", "out")] is not send

    comp2._status = Status.RUNNING
    current_manager = comp2._inner_manager
    comp2.start((recv,), (send,))
    assert comp2._inner_manager is current_manager


def test_graph_manager_additional_paths(monkeypatch) -> None:
    cleared: list[str] = []
    registered: list[dict[str, object]] = []
    fake_store = types.SimpleNamespace(
        clear_node=lambda node_id: cleared.append(node_id),
        register_thread=lambda **kwargs: registered.append(kwargs),
    )
    monkeypatch.setattr("src.core.graph.get_log_store", lambda: fake_store)
    monkeypatch.setattr("src.core.graph.time.sleep", lambda _s: None)
    monkeypatch.setattr(
        PrimitiveComponent,
        "registered_subclasses",
        classmethod(
            lambda cls: {"Known": _CompositeInner, "OutputOnly": _OutputOnlyComp}
        ),
    )

    gm = GraphManager(Graph(nodes={}, edges=[]))
    with pytest.raises(ValueError):
        gm.add_primitive_node("Missing", {})

    added_id, _ = gm.add_primitive_node("OutputOnly", {})
    assert gm.component(added_id).type_ == "_OutputOnlyComp"

    running_node = Node(id_="running", type="Known", init_args={})
    gm2 = GraphManager(Graph(nodes={"running": running_node}, edges=[]))
    gm2._components["running"] = types.SimpleNamespace(
        status=types.SimpleNamespace(value="running")
    )
    calls: list[str] = []
    gm2.stop = lambda: calls.append("stop")  # type: ignore[method-assign]
    gm2.run = lambda: calls.append("run")  # type: ignore[method-assign]
    updated = gm2.update_primitive_node_init_args("running", {})
    assert updated is running_node
    assert calls == ["stop", "run"]
    assert gm2.update_primitive_node_init_args("missing", {}) is None

    gm3 = GraphManager(
        Graph(
            nodes={"running": Node(id_="running", type="Missing", init_args={})},
            edges=[],
        )
    )
    monkeypatch.setattr(
        PrimitiveComponent,
        "registered_subclasses",
        classmethod(lambda cls: {"Known": _CompositeInner}),
    )
    assert gm3.update_primitive_node_init_args("running", {}) is None

    nodes = {
        "a": Node(id_="a", type="Known", init_args={}),
        "b": Node(id_="b", type="Known", init_args={}),
        "c": Node(id_="c", type="Known", init_args={}),
    }
    edges = [
        Edge(source_node="a", source_slot="out", target_node="b", target_slot="plain"),
        Edge(source_node="c", source_slot="out", target_node="a", target_slot="plain"),
    ]
    gm4 = GraphManager(Graph(nodes=nodes, edges=edges))
    gm4.delete_node("missing")
    gm4.delete_node("a")
    assert cleared == ["a"]
    assert "a" not in gm4.graph.nodes
    assert gm4.graph.edges == []
    assert gm4.components()["b"].stop_event.is_set() is True
    assert gm4.components()["c"].stop_event.is_set() is False  # upstream, not stopped

    composite_node = Node(
        id_="wrap",
        type="Wrap",
        init_args={},
        sub_graph=Graph(
            nodes={"n1": Node(id_="n1", type="Known", init_args={})}, edges=[]
        ),
    )
    gm5 = GraphManager(Graph(nodes={"wrap": composite_node}, edges=[]))
    assert isinstance(gm5.component("wrap"), CompositeComponent)

    ui_node = Node(id_="ui", type="Known", init_args={})
    gm6 = GraphManager(Graph(nodes={"ui": ui_node}, edges=[]))
    gm6._sender_handles.clear()
    gm6.run()
    assert ("ui", "ui_text") in gm6.ui_senders()
    assert ("ui", "ui_text") in gm6.ui_receivers()
    inner = gm6.component("ui")
    assert isinstance(inner, _CompositeInner)
    assert inner.started_inputs is not None and inner.started_inputs.maybe is None
    assert registered and registered[0]["node_id"] == "ui"

    assert GraphManager._build_tuple(None, {"x": 1}) == ()
    tuple_out = GraphManager._build_tuple(tuple[int, str], {"2": "b", "1": 1})
    assert tuple_out == (1, "b")


def test_frames_additional_paths(monkeypatch) -> None:
    pcm = np.array([0, 32767, -32768, 16384], dtype=np.int16).tobytes()
    stereo = AudioFrame.new(data=pcm, sample_rate=4, channels=2)
    assert stereo.data.shape == (2, 2)
    mono_from_bytes = AudioFrame.new(
        data=np.array([0, 1], dtype=np.int16).tobytes(), sample_rate=2, channels=1
    )
    assert mono_from_bytes.data.shape == (1, 2)

    mono = AudioFrame.new(
        data=np.array([0, 32767], dtype=np.int16),
        sample_rate=2,
        channels=1,
    )
    assert mono.data.shape == (1, 2)
    deinterleaved = AudioFrame.new(
        data=np.array([0, 1, 2, 3], dtype=np.int16),
        sample_rate=2,
        channels=2,
    )
    assert deinterleaved.data.shape == (2, 2)

    transposed = AudioFrame.new(
        data=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        sample_rate=3,
        channels=2,
    )
    assert transposed.data.shape == (2, 3)
    with pytest.raises(ValueError):
        AudioFrame.new(data="bad", sample_rate=1)  # type: ignore[arg-type]

    resampled = stereo.get(AudioDataFormat.FLOAT32, sample_rate=8, num_channels=1)
    assert resampled.shape == (1, 4)
    duplicated = mono.get(AudioDataFormat.FLOAT32, num_channels=2)
    assert duplicated.shape == (2, 2)
    reduced = stereo.get(AudioDataFormat.FLOAT32, num_channels=1)
    assert reduced.shape == (1, 2)
    padded = stereo.get(AudioDataFormat.FLOAT32, num_channels=3)
    assert padded.shape == (3, 2)
    three_channel = AudioFrame.new(
        data=np.zeros((3, 2), dtype=np.float32),
        sample_rate=2,
        channels=3,
    )
    trimmed = three_channel.get(AudioDataFormat.FLOAT32, num_channels=2)
    assert trimmed.shape == (2, 2)
    assert isinstance(stereo.get(AudioDataFormat.PCM16), bytes)
    assert isinstance(stereo.get(AudioDataFormat.PCM8), bytes)
    with pytest.raises(ValueError):
        stereo.get("bad")  # type: ignore[arg-type]

    request = RequestFrame.new()
    assert "RequestFrame" in str(request)
    text = TextFrame.new(text="hello")
    assert text.get() == "hello"
    interrupt = InterruptFrame.new(reason="x")
    assert interrupt.reason == "x"
    message = MessageFrame.new(
        role="assistant", content="ok", tool_calls=[], tool_call_id="1"
    )
    assert message.content == "ok"
    tool_def = ToolDef.new(name="n", description="d", parameters={"a": 1}, strict=True)
    assert tool_def.strict is True
    tool_call = ToolCall.new(call_id="c", name="n", arguments="{}")
    assert tool_call.arguments == "{}"
    tool_result = ToolResult.new(call_id="c", content="done")
    assert tool_result.content == "done"

    pose = BodyPoseFrame(poses={"head": BonePose(), "left_hand": None})
    assert pose.get()["head"] is not None
    assert "active_parts=1/2" in str(pose)

    boxes = np.zeros((1, 1, 4), dtype=np.float32)
    scores = np.ones((1, 1), dtype=np.float32)
    prompts = ("cat",)
    detected = ObjectDetectionFrame.new(boxes=boxes, scores=scores, prompts=prompts)
    assert detected.prompts == prompts
    masks = np.zeros((1, 2, 2), dtype=bool)
    segmented = ObjectSegmentationFrame.new(
        masks=masks,
        boxes=np.zeros((1, 4), dtype=np.float32),
        scores=np.ones((1,), dtype=np.float32),
        object_ids=np.array([1], dtype=np.int64),
        labels=("cat",),
    )
    assert segmented.labels == ("cat",)
    located = ObjectLocationFrame.new(
        labels=("cat",),
        positions=np.zeros((1, 3), dtype=np.float32),
        depths=np.ones((1,), dtype=np.float32),
        scores=np.ones((1,), dtype=np.float32),
        boxes=np.zeros((1, 4), dtype=np.float32),
        object_ids=np.array([1], dtype=np.int64),
    )
    assert located.labels == ("cat",)
    goal = GoalFrame.new(x=1.0, y=2.0, z=3.0)
    assert goal.z == 3.0
    camera = CameraParamsFrame.new(
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
        width=2,
        height=3,
    )
    assert camera.width == 2
    depth = DepthFrame.new(data=np.zeros((3, 4), dtype=np.float32), is_metric=True)
    assert depth.width == 4 and depth.height == 3 and depth.is_metric is True

    fake_cv2 = types.SimpleNamespace(
        COLOR_BGR2RGB=1,
        COLOR_RGB2BGR=2,
        cvtColor=lambda data, code: data + code,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    video = VideoFrame.new(
        data=np.zeros((2, 3, 3), dtype=np.uint8), format=VideoDataFormat.BGR
    )
    assert video.get(VideoDataFormat.BGR).shape == (2, 3, 3)
    assert np.all(video.get(VideoDataFormat.RGB) == 1)

    stereo_video = StereoVideoFrame.new(
        left=np.zeros((2, 2, 3), dtype=np.uint8),
        right=np.ones((2, 2, 3), dtype=np.uint8),
        format=VideoDataFormat.RGB,
    )
    assert np.all(stereo_video.get("left", VideoDataFormat.RGB) == 0)
    assert np.all(stereo_video.get("right", VideoDataFormat.BGR) == 3)

    stereo_camera = StereoCameraParamsFrame.new(
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
        baseline=0.1,
        width=2,
        height=2,
    )
    assert stereo_camera.baseline == 0.1
