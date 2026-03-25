from __future__ import annotations

import json
from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import EmitOnStart, ThreadedComponent, Tag
from src.core.frames import GoalFrame, TextFrame, ToolCall, ToolDef, ToolResult


class MovementToolInputs(NamedTuple):
    tool_call: Receiver[ToolCall]


class MovementToolOutputs(NamedTuple):
    tool_def: Sender[ToolDef]
    tool_result: Sender[ToolResult]
    goal: Sender[GoalFrame]
    instruction: Sender[TextFrame]


class MovementTool(
    ThreadedComponent[MovementToolInputs, MovementToolOutputs],
    EmitOnStart[MovementToolOutputs],
):
    description = "Translates LLM tool calls into movement commands for DartControl"
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def emit(self, outputs: MovementToolOutputs) -> None:
        outputs.tool_def.send(
            ToolDef.new(
                name="move",
                description="You are an embodied agent with a physical body. Use this tool to perform motions. "
                            "YOU MUST Provide *BOTH* x and y when you want to move to a specific location. "
                            "Omit x and y for in-place motions like dancing or waving.",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Target x position in meters"},
                        "z": {"type": "number", "description": "Target z position in meters"},
                        "instruction": {
                            "type": "string",
                            "description": "Any motion, e.g. 'walk', 'run', 'sit', 'wave', 'dance'",
                        },
                    },
                    "required": ["instruction"],
                },
            )
        )

    def run(self, inputs: MovementToolInputs, outputs: MovementToolOutputs) -> None:
        for call in inputs.tool_call(self):
            if call is None:
                break
            if call.name != "move":
                outputs.tool_result.send(
                    ToolResult.new(call_id=call.call_id, content="unknown tool")
                )
                continue

            args = json.loads(call.arguments)
            instruction = args["instruction"]
            x = args.get("x")
            z = args.get("z")

            outputs.instruction.send(TextFrame.new(text=instruction))
            if x is not None and z is not None:
                outputs.goal.send(GoalFrame.new(x=float(x), y=0.0, z=float(z)))

            result = f"Executing '{instruction}'"
            if x is not None and z is not None:
                result += f" toward ({x}, {z})"
            outputs.tool_result.send(
                ToolResult.new(call_id=call.call_id, content=result)
            )
