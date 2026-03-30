from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import ToolCall, ToolDef, ToolResult


class DoNothingToolConfig(BaseModel):
    tool_description: str = "Do nothing. Call this when no action is needed."


class DoNothingToolInputs(NamedTuple):
    tool_call: Receiver[ToolCall]


class DoNothingToolOutputs(NamedTuple):
    tool_def: Sender[ToolDef]
    tool_result: Sender[ToolResult]


class DoNothingTool(
    ThreadedComponent[DoNothingToolInputs, DoNothingToolOutputs],
):
    description = "A do-nothing tool: emits its definition and returns empty results"
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: DoNothingToolConfig = DoNothingToolConfig()) -> None:
        super().__init__()
        self.config = config

    def setup(self, outputs: DoNothingToolOutputs) -> None:
        outputs.tool_def.send(
            ToolDef.new(
                name="do_nothing",
                description=self.config.tool_description,
                parameters={"type": "object", "properties": {}, "required": []},
            )
        )

    def run(self, inputs: DoNothingToolInputs, outputs: DoNothingToolOutputs) -> None:
        for call in inputs.tool_call:
            if call is None:
                break
            outputs.tool_result.send(ToolResult.new(call_id=call.call_id, content=""))
