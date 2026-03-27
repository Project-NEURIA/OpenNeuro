from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.core.conduit.agent_state import AgentState, AgentStateConfig, AgentStateInputs, AgentStateOutputs
from src.core.frames import BodyPoseFrame, BonePose, MessageFrame, ObjectLocationFrame, TextFrame, ToolCall, ToolResult


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def test_agent_state_helpers(capsys) -> None:
    state = AgentState(AgentStateConfig(system_prompt="system"))
    assert state._build_visible_message() is None

    first = ObjectLocationFrame.new(
        labels=("cup",),
        positions=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        depths=np.array([1.0], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        boxes=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        object_ids=np.array([1], dtype=np.int64),
    )
    state._diff_objects(first)
    visible = state._build_visible_message()
    assert visible is not None
    assert "cup" in visible.content

    second = ObjectLocationFrame.new(
        labels=("book",),
        positions=np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        depths=np.array([2.0], dtype=np.float32),
        scores=np.array([0.8], dtype=np.float32),
        boxes=np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
        object_ids=np.array([2], dtype=np.int64),
    )
    state._diff_objects(second)
    assert any("last seen" in (msg.content or "") for msg in state._history)
    assert state._heading_from_quat(1.0, 0.0, 0.0, 0.0) == 0.0

    tool_call = ToolCall.new(call_id="call-1", name="lookup", arguments="{}")
    messages = [
        MessageFrame.new(role="assistant", content="tool", tool_calls=[tool_call]),
        MessageFrame.new(role="tool", content="done", tool_call_id="call-1"),
        MessageFrame.new(role="system", content=None),
    ]
    state._print_messages(messages)
    printed = capsys.readouterr().out
    assert "tool_calls=['lookup']" in printed
    assert "tool_call_id=call-1" in printed
    assert "(no content)" in printed


def test_agent_state_run(monkeypatch) -> None:
    state = AgentState(AgentStateConfig(system_prompt="base system"))
    sent_messages = []

    request = TextFrame.new(text="request text")
    speech = TextFrame.new(text="spoken input")
    feedback = TextFrame.new(text="assistant reply")
    vision = TextFrame.new(text="vision note")
    memory = TextFrame.new(text="memory note")
    tool_call = ToolCall.new(call_id="tool-1", name="lookup", arguments='{"q":"x"}')
    tool_result = ToolResult.new(call_id="tool-1", content="tool output")

    monkeypatch.setattr(
        "src.core.conduit.agent_state.drain",
        lambda *args: [(speech, feedback, vision, memory, tool_call, tool_result)],
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
        request=_FakeRecv([request, None]),
        initial_msgs=_FakeRecv([[MessageFrame.new(role="system", content="init")]]),
        speech=_FakeRecv([speech]),
        feedback=_FakeRecv([feedback]),
        tool_call=_FakeRecv([tool_call]),
        tool_result=_FakeRecv([tool_result]),
        vision=_FakeRecv([vision]),
        pose=_FakeRecv([pose_frame]),
        objects=_FakeRecv([object_frame]),
        memory=_FakeRecv([memory]),
    )
    outputs = AgentStateOutputs(
        messages=SimpleNamespace(send=lambda value: sent_messages.append(value))
    )

    state.run(inputs, outputs)

    assert len(sent_messages) == 1
    msgs = sent_messages[0]
    contents = [msg.content for msg in msgs]
    assert contents[0] == "init"
    assert any("request text" in (content or "") for content in contents)
    assert any("spoken input" in (content or "") for content in contents)
    assert any("assistant reply" in (content or "") for content in contents)
    assert any("vision note" in (content or "") for content in contents)
    assert any("memory note" in (content or "") for content in contents)
    assert any("Currently visible objects" in (content or "") for content in contents)
    assert any("Heading (from +Z clockwise): -0" in (content or "") for content in contents)
    assert any(msg.tool_calls and msg.tool_calls[0].name == "lookup" for msg in msgs)
    assert any(msg.tool_call_id == "tool-1" and "tool output" in (msg.content or "") for msg in msgs)
