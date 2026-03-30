from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import TextFrame, ToolCall, ToolDef, ToolResult


class StopToolConfig(BaseModel):
    tool_description: str = (
        "Stop the current reasoning loop. Call this when you have finished "
        "all actions and have nothing more to do. This pauses the agent "
        "until new user input arrives."
    )


class StopToolInputs(NamedTuple):
    tool_call: Receiver[ToolCall]


class StopToolOutputs(NamedTuple):
    tool_def: Sender[ToolDef]
    tool_result: Sender[ToolResult]
    stop: Sender[TextFrame]


class StopTool(
    ThreadedComponent[StopToolInputs, StopToolOutputs],
):
    description = "Stops the agent reasoning loop until new input arrives"
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: StopToolConfig) -> None:
        super().__init__()
        self.config = config

    def setup(self, outputs: StopToolOutputs) -> None:
        outputs.tool_def.send(
            ToolDef.new(
                name="stop",
                description=self.config.tool_description,
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )
        )

    def run(self, inputs: StopToolInputs, outputs: StopToolOutputs) -> None:
        for call in inputs.tool_call:
            if call is None:
                break
            if call.name != "stop":
                continue
            print("[Stop] Agent stopping loop")
            outputs.tool_result.send(
                ToolResult.new(call_id=call.call_id, content="")
            )
            outputs.stop.send(TextFrame.new(text="stop"))
