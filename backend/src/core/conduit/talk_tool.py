from __future__ import annotations

import json
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import TextFrame, ToolCall, ToolDef, ToolResult


class TalkToolConfig(BaseModel):
    tool_description: str = (
        "Say something out loud to the user. The text will be spoken via TTS."
    )


class TalkToolInputs(NamedTuple):
    tool_call: Receiver[ToolCall]


class TalkToolOutputs(NamedTuple):
    tool_def: Sender[ToolDef]
    tool_result: Sender[ToolResult]
    text: Sender[TextFrame]


class TalkTool(
    ThreadedComponent[TalkToolInputs, TalkToolOutputs],
):
    description = "Lets the agent speak by sending text to TTS"
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: TalkToolConfig) -> None:
        super().__init__()
        self.config = config

    def setup(self, outputs: TalkToolOutputs) -> None:
        outputs.tool_def.send(
            ToolDef.new(
                name="talk",
                description=self.config.tool_description,
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to speak.",
                        },
                    },
                    "required": ["text"],
                },
            )
        )

    def run(self, inputs: TalkToolInputs, outputs: TalkToolOutputs) -> None:
        for call in inputs.tool_call:
            if call is None:
                break
            if call.name != "talk":
                continue
            try:
                text = json.loads(call.arguments).get("text", "")
            except (json.JSONDecodeError, AttributeError):
                text = call.arguments
            if text:
                print(f"[Talk] {text}")
                outputs.text.send(TextFrame.new(text=text))
            outputs.tool_result.send(
                ToolResult.new(call_id=call.call_id, content="")
            )
