from __future__ import annotations

import json
from typing import NamedTuple

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import ToolCall, ToolDef, ToolResult


class ScratchpadToolInputs(NamedTuple):
    tool_call: Receiver[ToolCall]


class ScratchpadToolOutputs(NamedTuple):
    tool_def: Sender[ToolDef]
    tool_result: Sender[ToolResult]


class ScratchpadTool(ThreadedComponent[ScratchpadToolInputs, ScratchpadToolOutputs]):
    description = "Persistent scratchpad the agent can **read** and **update** to maintain working notes across turns."
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self) -> None:
        super().__init__()
        self._content: str = ""

    def setup(self, outputs: ScratchpadToolOutputs) -> None:
        outputs.tool_def.send(
            ToolDef.new(
                name="scratchpad_update",
                description="Overwrite the scratchpad with new content. Use this to keep track of plans, observations, or any working notes you want to persist across turns.",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The new scratchpad content (replaces everything).",
                        },
                    },
                    "required": ["content"],
                },
            )
        )
        outputs.tool_def.send(
            ToolDef.new(
                name="scratchpad_read",
                description="Read the current scratchpad content.",
                parameters={
                    "type": "object",
                    "properties": {},
                },
            )
        )

    def run(self, inputs: ScratchpadToolInputs, outputs: ScratchpadToolOutputs) -> None:
        for call in inputs.tool_call:
            if call is None:
                break

            if call.name == "scratchpad_update":
                try:
                    args = json.loads(call.arguments)
                    self._content = args.get("content", "")
                except (json.JSONDecodeError, AttributeError):
                    self._content = call.arguments
                print(f"[Scratchpad] Updated: {self._content[:80]}")
                outputs.tool_result.send(
                    ToolResult.new(call_id=call.call_id, content="ok")
                )

            elif call.name == "scratchpad_read":
                print(f"[Scratchpad] Read: {self._content[:80]}")
                outputs.tool_result.send(
                    ToolResult.new(
                        call_id=call.call_id,
                        content=self._content or "(empty)",
                    )
                )
