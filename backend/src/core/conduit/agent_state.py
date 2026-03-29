from __future__ import annotations

import numpy as np
import math
from datetime import datetime
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import (
    BodyPoseFrame,
    MessageFrame,
    ObjectLocationFrame,
    TextFrame,
    ToolCall,
    ToolResult,
)
from src.core.utils import drain


class AgentStateConfig(BaseModel):
    system_prompt: str


class AgentStateInputs[T](NamedTuple):
    request: Receiver[T]
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


class AgentState[T](ThreadedComponent[AgentStateInputs[T], AgentStateOutputs]):
    """Manages conversation history, optionally enriched by memory and character card."""

    description = "Tracks and manages **agent conversation state**. Maintains message history enriched by optional *memory* and *character card* inputs, and outputs assembled `MessageFrame` lists for the LLM."
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
                msg = MessageFrame.new(
                    role="system",
                    content=f'[Object "{label}" (id={oid}) last seen at ({x:.2f}, {y:.2f}, {z:.2f})]',
                )
                self._history.append(msg)
                self._print_message(msg)

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

    @staticmethod
    def _heading_from_quat(w: float, x: float, y: float, z: float) -> float:
        """Extract heading in degrees (clockwise from +Z) from a Y-up quaternion."""
        fwd_x = 2 * (x * z + w * y)
        fwd_z = 1 - 2 * (x * x + y * y)
        return -math.degrees(math.atan2(fwd_x, fwd_z))

    @staticmethod
    def _print_message(m: MessageFrame) -> None:
        preview = m.content[:120] if m.content else "(no content)"
        extra = ""
        if m.tool_calls:
            for tc in m.tool_calls:
                args = tc.arguments[:80] if tc.arguments else ""
                extra += f" {tc.name}({args})"
        if m.tool_call_id:
            extra += f" tool_call_id={m.tool_call_id}"
        print(f"  [{m.role}] {preview}{extra}")

    def run(self, inputs: AgentStateInputs, outputs: AgentStateOutputs) -> None:
        print("[AgentState] Starting Agent State management")

        # Read initial prompts once (constant component, e.g. CharacterCard)
        if inputs.initial_msgs is not None:
            frame = next(inputs.initial_msgs)
            if frame is not None:
                self._history = frame + self._history
                print(f"[AgentState] Initial messages loaded ({len(frame)} msgs)")

        # Configure receiver modes for optional inputs
        if inputs.speech:
            inputs.speech.blocking = False
        if inputs.feedback:
            inputs.feedback.blocking = False
        if inputs.vision:
            inputs.vision.blocking = False
        if inputs.memory:
            inputs.memory.blocking = False
        if inputs.tool_call:
            inputs.tool_call.blocking = False
        if inputs.tool_result:
            inputs.tool_result.blocking = False
        if inputs.objects is not None:
            inputs.objects.newest = True
            inputs.objects.blocking = False
        if inputs.pose is not None:
            inputs.pose.newest = True
            inputs.pose.blocking = False

        # Buffer tool_calls until their matching tool_result arrives
        pending_tool_calls: dict[str, ToolCall] = {}

        # Block on request, drain others on each trigger
        for req in inputs.request:
            if req is None:
                break

            # Drain everything except tool results
            for speech, feedback, vision, memory, tc in drain(
                inputs.speech,
                inputs.feedback,
                inputs.vision,
                inputs.memory,
                inputs.tool_call,
            ):
                if speech is not None:
                    ts = datetime.fromtimestamp(speech.pts / 1e9).strftime("%H:%M:%S")
                    msg = MessageFrame.new(role="user", content=f"[{ts}] {speech.text}")
                    self._history.append(msg)
                    self._print_message(msg)
                if feedback is not None:
                    ts = datetime.fromtimestamp(feedback.pts / 1e9).strftime("%H:%M:%S")
                    msg = MessageFrame.new(
                        role="assistant", content=f"[{ts}] {feedback.text}"
                    )
                    self._history.append(msg)
                    self._print_message(msg)
                if vision is not None:
                    ts = datetime.fromtimestamp(vision.pts / 1e9).strftime("%H:%M:%S")
                    msg = MessageFrame.new(role="system", content=f"[{ts}] {vision.text}")
                    self._history.append(msg)
                    self._print_message(msg)
                if memory is not None:
                    ts = datetime.fromtimestamp(memory.pts / 1e9).strftime("%H:%M:%S")
                    msg = MessageFrame.new(role="system", content=f"[{ts}] {memory.text}")
                    self._history.append(msg)
                    self._print_message(msg)
                if tc is not None:
                    # Tool call in chronological position + placeholder result
                    ts = datetime.fromtimestamp(tc.pts / 1e9).strftime("%H:%M:%S")
                    msg_tc = MessageFrame.new(
                        role="assistant",
                        content=f"[{ts}]",
                        tool_calls=[tc],
                    )
                    self._history.append(msg_tc)
                    self._print_message(msg_tc)
                    msg_tr = MessageFrame.new(
                        role="tool",
                        content="(pending)",
                        tool_call_id=tc.call_id,
                    )
                    self._history.append(msg_tr)
                    pending_tool_calls[tc.call_id] = tc

            # Drain tool results and replace placeholders
            if inputs.tool_result is not None:
                for tr in inputs.tool_result:
                    if tr is None:
                        break
                    ts = datetime.fromtimestamp(tr.pts / 1e9).strftime("%H:%M:%S")
                    for i, m in enumerate(self._history):
                        if m.tool_call_id == tr.call_id and m.content == "(pending)":
                            self._history[i] = MessageFrame.new(
                                role="tool",
                                content=f"[{ts}] {tr.content}",
                                tool_call_id=tr.call_id,
                            )
                            self._print_message(self._history[i])
                            pending_tool_calls.pop(tr.call_id, None)
                            break

            # Diff objects against previous state
            if inputs.objects is not None:
                obj_frame = next(inputs.objects)
                if obj_frame is not None:
                    self._diff_objects(obj_frame)

            # Build final messages: history
            msgs = self._history.copy()

            # visible objects
            visible_msg = self._build_visible_message()
            if visible_msg is not None:
                msgs.append(visible_msg)

            # Read agent spatial state (position + direction)
            if inputs.pose is not None:
                pose_frame = next(inputs.pose)
                if pose_frame is not None:
                    poses = pose_frame.get()
                    waist = poses.get("waist")
                    if waist is not None:
                        px, py, pz = waist.pos_x, waist.pos_y, waist.pos_z
                        heading = self._heading_from_quat(
                            waist.rot_w, waist.rot_x, waist.rot_y, waist.rot_z
                        )
                        msgs.append(
                            MessageFrame.new(
                                role="system",
                                content=f"[Position (y-up): x={px:.2f}, y={py:.2f}, z={pz:.2f}]\n"
                                f"[Heading (from +Z clockwise): {heading:.0f}°]",
                            )
                        )

            outputs.messages.send(msgs)

        print("[AgentState] Agent State management stopped")
