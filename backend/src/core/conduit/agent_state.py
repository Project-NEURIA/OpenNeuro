from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import MessageFrame, RequestFrame, TextFrame, ToolCall, ToolResult
from src.core.utils import drain


class AgentStateConfig(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."


class AgentStateInputs(NamedTuple):
    request: Receiver[RequestFrame]
    initial_msgs: Receiver[list[MessageFrame]] | None = None
    speech: Receiver[TextFrame] | None = None
    feedback: Receiver[TextFrame] | None = None
    tool_call: Receiver[ToolCall] | None = None
    tool_result: Receiver[ToolResult] | None = None
    vision: Receiver[TextFrame] | None = None
    memory: Receiver[TextFrame] | None = None


class AgentStateOutputs(NamedTuple):
    messages: Sender[list[MessageFrame]]
    # messages_for_memory: Sender[list[MessageFrame]] | None = None


class AgentState(ThreadedComponent[AgentStateInputs, AgentStateOutputs]):
    """Manages conversation history, optionally enriched by memory and character card."""

    description = "Tracks and manages agent conversation state"
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: AgentStateConfig) -> None:
        super().__init__()
        self.config = config
        self._history: list[MessageFrame] = [
            MessageFrame.new(role="system", content=config.system_prompt)
        ]

    def run(self, inputs: AgentStateInputs, outputs: AgentStateOutputs) -> None:
        print("[AgentState] Starting Agent State management")

        # Read initial prompts once (constant component, e.g. CharacterCard)
        if inputs.initial_msgs is not None:
            frame = next(inputs.initial_msgs(self))
            if frame is not None:
                self._history = frame + self._history
                print(f"[AgentState] Initial messages loaded ({len(frame)} msgs)")

        # Block on request, drain others on each trigger
        for req in inputs.request(self):
            if req is None:
                break

            for speech, feedback, vision, memory in drain(
                self, inputs.speech, inputs.feedback, inputs.vision, inputs.memory
            ):
                if speech is not None:
                    self._history.append(
                        MessageFrame.new(role="user", content=speech.text)
                    )
                if feedback is not None:
                    self._history.append(
                        MessageFrame.new(role="assistant", content=feedback.text)
                    )
                if vision is not None:
                    self._history.append(
                        MessageFrame.new(role="system", content=vision.text)
                    )
                if memory is not None:
                    self._history.append(
                        MessageFrame.new(role="system", content=memory.text)
                    )

            outputs.messages.send(self._history.copy())

        print("[AgentState] Agent State management stopped")
