from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.core.conduit.agent_state import (
    AgentState,
    AgentStateConfig,
    AgentStateInputs,
    AgentStateOutputs,
)
from src.core.frames import (
    BodyPoseFrame,
    BonePose,
    MessageFrame,
    ObjectLocationFrame,
    TextFrame,
    ToolCall,
    ToolResult,
)


class _FakeReceiver:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self.blocking = True
        self.newest = False

    def __iter__(self) -> _FakeReceiver:
        return self

    def __next__(self) -> object:
        if self._items:
            return self._items.pop(0)
        return None


def test_agent_state_helper_methods(capsys) -> None:
    state = AgentState(AgentStateConfig(system_prompt="system"))

    # _heading_from_quat: identity quaternion => 0 degrees
    assert state._heading_from_quat(1.0, 0.0, 0.0, 0.0) == 0.0

    # _print_message: tool call, tool result, empty content
    tool_call = ToolCall.new(call_id="call-1", name="lookup", arguments='{"q":"x"}')
    tool_msg = MessageFrame.new(
        role="assistant", content="tool", tool_calls=[tool_call]
    )
    result_msg = MessageFrame.new(role="tool", content="done", tool_call_id="call-1")
    empty_msg = MessageFrame.new(role="system", content=None)

    state._print_message(tool_msg)
    state._print_message(result_msg)
    state._print_message(empty_msg)

    printed = capsys.readouterr().out
    assert 'lookup({"q":"x"})' in printed
    assert "tool_call_id=call-1" in printed
    assert "(no content)" in printed


def test_agent_state_run_builds_messages_and_dumps_json(monkeypatch) -> None:
    state = AgentState(AgentStateConfig(system_prompt="base system"))
    sent_messages: list[list[MessageFrame]] = []

    request = TextFrame.new(text="request text")
    speech = TextFrame.new(text="spoken input")
    feedback = TextFrame.new(text="assistant reply")
    vision = TextFrame.new(text="vision note")
    memory = TextFrame.new(text="memory note")
    tool_call = ToolCall.new(call_id="tool-1", name="lookup", arguments='{"q":"x"}')
    tool_result = ToolResult.new(call_id="tool-1", content="tool output")

    monkeypatch.setattr(
        "src.core.conduit.agent_state.drain",
        lambda *args: [(speech, feedback, memory, tool_call)],
    )
    projects_dir = Path("tests_runtime") / "agent_state"
    project_dir = projects_dir / "demo-project"
    project_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.core.conduit.agent_state.PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(
        "src.core.conduit.agent_state.AppConfig.load_config",
        lambda: SimpleNamespace(current_project="demo-project"),
    )
    object_frame = ObjectLocationFrame.new(
        labels=("lamp",),
        positions=np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
        depths=np.array([3.0], dtype=np.float32),
        scores=np.array([0.95], dtype=np.float32),
        boxes=np.array([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32),
        object_ids=np.array([10], dtype=np.int64),
    )
    pose_frame = BodyPoseFrame(
        poses={"waist": BonePose(pos_x=1.0, pos_y=2.0, pos_z=3.0, rot_w=1.0)}
    )

    inputs = AgentStateInputs(
        request=_FakeReceiver([request, None]),
        initial_msgs=_FakeReceiver([[MessageFrame.new(role="system", content="init")]]),
        speech=_FakeReceiver([]),
        feedback=_FakeReceiver([]),
        tool_call=_FakeReceiver([]),
        tool_result=_FakeReceiver([tool_result, None]),
        vision=_FakeReceiver([vision]),
        pose=_FakeReceiver([pose_frame]),
        objects=_FakeReceiver([object_frame]),
        memory=_FakeReceiver([]),
    )
    outputs = AgentStateOutputs(
        messages=SimpleNamespace(send=lambda value: sent_messages.append(value))
    )

    state.run(inputs, outputs)

    assert len(sent_messages) == 1
    msgs = sent_messages[0]
    contents = [msg.content for msg in msgs]
    assert contents[0] == "init"
    assert any("spoken input" in (content or "") for content in contents)
    assert any("assistant reply" in (content or "") for content in contents)
    assert any("vision note" in (content or "") for content in contents)
    assert any("memory note" in (content or "") for content in contents)
    assert any("Currently visible objects" in (content or "") for content in contents)
    assert any(
        "Heading (from +Z clockwise): -0" in (content or "") for content in contents
    )
    assert any(msg.tool_calls and msg.tool_calls[0].name == "lookup" for msg in msgs)
    assert any(
        msg.tool_call_id == "tool-1" and "tool output" in (msg.content or "")
        for msg in msgs
    )

    dumped = json.loads((project_dir / "messages.json").read_text())
    assert dumped[0]["content"] == "init"
    assert any(entry.get("tool_calls") for entry in dumped)
    assert any(entry.get("tool_call_id") == "tool-1" for entry in dumped)


def test_agent_state_dump_messages_without_current_project(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.core.conduit.agent_state.AppConfig.load_config",
        lambda: SimpleNamespace(current_project=None),
    )

    AgentState._dump_messages([MessageFrame.new(role="user", content="hello")])
