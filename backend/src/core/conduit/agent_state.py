from __future__ import annotations

import numpy as np
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import (
    BodyPoseFrame,
    MessageFrame,
    ObjectLocationFrame,
    RequestFrame,
    TextFrame,
    ToolCall,
    ToolResult,
)
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
    pose: Receiver[BodyPoseFrame] | None = None
    objects: Receiver[ObjectLocationFrame] | None = None
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
        # Object tracking: object_id -> (label, position)
        self._visible: dict[int, tuple[str, np.ndarray]] = {}

    def _diff_objects(self, frame: ObjectLocationFrame) -> None:
        """Diff incoming objects against currently visible. Disappeared objects
        get frozen into history, visible objects update in-place."""
        new_visible: dict[int, tuple[str, np.ndarray]] = {}
        for i, obj_id in enumerate(frame.object_ids):
            oid = int(obj_id)
            new_visible[oid] = (frame.labels[i], frame.positions[i])

        # Disappeared objects -> freeze into history
        for oid, (label, pos) in self._visible.items():
            if oid not in new_visible:
                x, y, z = pos
                self._history.append(
                    MessageFrame.new(
                        role="system",
                        content=f'[Object "{label}" (id={oid}) last seen at ({x:.2f}, {y:.2f}, {z:.2f})]',
                    )
                )

        self._visible = new_visible

    def _build_visible_message(self) -> MessageFrame | None:
        """Build a live message describing currently visible objects."""
        if not self._visible:
            return None
        lines = []
        for oid, (label, pos) in self._visible.items():
            x, y, z = pos
            lines.append(f'  "{label}" (id={oid}) at ({x:.2f}, {y:.2f}, {z:.2f})')
        return MessageFrame.new(
            role="system",
            content="[Currently visible objects]\n" + "\n".join(lines),
        )

    def run(self, inputs: AgentStateInputs, outputs: AgentStateOutputs) -> None:
        print("[AgentState] Starting Agent State management")

        # Read initial prompts once (constant component, e.g. CharacterCard)
        if inputs.initial_msgs is not None:
            frame = next(inputs.initial_msgs(self))
            if frame is not None:
                self._history = frame + self._history
                print(f"[AgentState] Initial messages loaded ({len(frame)} msgs)")

        # Newest-only iterator for objects (high frequency, only latest matters)
        objects_it = (
            inputs.objects(self, newest=True)
            if inputs.objects is not None
            else None
        )

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

            # Diff objects against previous state
            if objects_it is not None:
                obj_frame = next(objects_it)
                if obj_frame is not None:
                    self._diff_objects(obj_frame)

            # Build final messages: history + live visible objects at the end
            msgs = self._history.copy()
            visible_msg = self._build_visible_message()
            if visible_msg is not None:
                msgs.append(visible_msg)

            outputs.messages.send(msgs)

        print("[AgentState] Agent State management stopped")
