from __future__ import annotations

from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import EmitOnStart, ThreadedComponent, Tag
from src.core.frames import ToolCall, ToolDef, ToolResult


class ThinkToolInputs(NamedTuple):
    tool_call: Receiver[ToolCall]


class ThinkToolOutputs(NamedTuple):
    tool_def: Sender[ToolDef]
    tool_result: Sender[ToolResult]


class ThinkTool(
    ThreadedComponent[ThinkToolInputs, ThinkToolOutputs],
    EmitOnStart[ThinkToolOutputs],
):
    description = "A scratchpad tool for the agent to think, plan, or reflect"
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def emit(self, outputs: ThinkToolOutputs) -> None:
        outputs.tool_def.send(
            ToolDef.new(
                name="think",
                description="Use this as a scratchpad to think, plan, or reflect before acting. "
                "Also use this when you have nothing else to do — you must always call a tool. "
                "Your thought is not shown to the user. Keep it brief — a few sentences, not paragraphs.",
                parameters={
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your brief internal thought or plan. Keep it short.",
                        },
                    },
                    "required": ["thought"],
                },
            )
        )

    def run(self, inputs: ThinkToolInputs, outputs: ThinkToolOutputs) -> None:
        import json

        for call in inputs.tool_call:
            if call is None:
                break
            if call.name != "think":
                continue
            try:
                thought = json.loads(call.arguments).get("thought", "")
            except (json.JSONDecodeError, AttributeError):
                thought = call.arguments
            print(f"[Think] {thought}")
            outputs.tool_result.send(ToolResult.new(call_id=call.call_id, content=""))
