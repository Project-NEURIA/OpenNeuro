from __future__ import annotations

import types

from src.core.conduit.think_tool import ThinkTool, ThinkToolInputs, ThinkToolOutputs
from src.core.conduit.throttle import (
    Throttle,
    ThrottleConfig,
    ThrottleInputs,
    ThrottleOutputs,
)
from src.core.frames import ToolCall


class _FakeRecv:
    def __init__(self, items):
        self._items = list(items)
        self._iter = iter(self._items)
        self.newest = False
        self.blocking = True

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


def test_think_tool_setup_and_run_paths(capsys) -> None:
    tool_defs = []
    tool_results = []
    tool = ThinkTool()

    outputs = ThinkToolOutputs(
        tool_def=types.SimpleNamespace(send=lambda value: tool_defs.append(value)),
        tool_result=types.SimpleNamespace(
            send=lambda value: tool_results.append(value)
        ),
    )

    tool.setup(outputs)
    tool.run(
        ThinkToolInputs(
            tool_call=_FakeRecv(
                [
                    ToolCall.new(call_id="skip", name="other", arguments="{}"),
                    ToolCall.new(
                        call_id="json", name="think", arguments='{"thought":"plan"}'
                    ),
                    ToolCall.new(
                        call_id="fallback", name="think", arguments="not-json"
                    ),
                    None,
                ]
            )
        ),
        outputs,
    )

    printed = capsys.readouterr().out
    assert tool_defs[0].name == "think"
    assert "[Think] plan" in printed
    assert "[Think] not-json" in printed
    assert [result.call_id for result in tool_results] == ["json", "fallback"]
    assert all(result.content == "" for result in tool_results)


def test_throttle_forwards_newest_items(monkeypatch) -> None:
    sleeps = []
    sent = []
    throttle = Throttle[int](ThrottleConfig(interval=0.25))

    monkeypatch.setattr(
        "src.core.conduit.throttle.time.sleep", lambda value: sleeps.append(value)
    )

    recv = _FakeRecv([1, 2, None])
    throttle.run(
        ThrottleInputs(data=recv),
        ThrottleOutputs(
            data=types.SimpleNamespace(send=lambda value: sent.append(value))
        ),
    )

    assert recv.newest is True
    assert sent == [1, 2]
    assert sleeps == [0.25, 0.25]
