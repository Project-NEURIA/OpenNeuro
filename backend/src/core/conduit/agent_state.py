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

    @staticmethod
    def _heading_from_quat(w: float, x: float, y: float, z: float) -> float:
        """Extract heading in degrees (clockwise from +Z) from a Y-up quaternion."""
        fwd_x = 2 * (x * z + w * y)
        fwd_z = 1 - 2 * (x * x + y * y)
        return -math.degrees(math.atan2(fwd_x, fwd_z))

    @staticmethod
    def _print_messages(msgs: list[MessageFrame]) -> None:
        print("-------------------------------------------")
        for m in msgs:
            preview = m.content[:80] if m.content else "(no content)"
            extra = ""
            if m.tool_calls:
                extra += f" tool_calls={[tc.name for tc in m.tool_calls]}"
            if m.tool_call_id:
                extra += f" tool_call_id={m.tool_call_id}"
            print(f"  [{m.role}] {preview}{extra}")

    def run(self, inputs: AgentStateInputs, outputs: AgentStateOutputs) -> None:
        print("[AgentState] Starting Agent State management")

        # Read initial prompts once (constant component, e.g. CharacterCard)
        if inputs.initial_msgs is not None:
            frame = next(inputs.initial_msgs(self, latest=False))
            if frame is not None:
                self._history = frame + self._history
                print(f"[AgentState] Initial messages loaded ({len(frame)} msgs)")

        # Upfront iterators (cursors persist across drains)
        speech_it = inputs.speech(self, no_block=True) if inputs.speech else None
        feedback_it = inputs.feedback(self, no_block=True) if inputs.feedback else None
        vision_it = inputs.vision(self, no_block=True) if inputs.vision else None
        memory_it = inputs.memory(self, no_block=True) if inputs.memory else None
        tool_call_it = inputs.tool_call(self, no_block=True) if inputs.tool_call else None
        tool_result_it = inputs.tool_result(self, no_block=True) if inputs.tool_result else None
        objects_it = (
            inputs.objects(self, newest=True, no_block=True)
            if inputs.objects is not None
            else None
        )
        pose_it = (
            inputs.pose(self, newest=True, no_block=True)
            if inputs.pose is not None
            else None
        )

        # Block on request, drain others on each trigger
        for req in inputs.request(self):
            if req is None:
                break

            for speech, feedback, vision, memory, tc, tr in drain(
                speech_it, feedback_it, vision_it, memory_it, tool_call_it, tool_result_it
            ):
                if speech is not None:
                    ts = datetime.fromtimestamp(speech.pts / 1e9).strftime("%H:%M:%S")
                    self._history.append(
                        MessageFrame.new(role="user", content=f"[{ts}] {speech.text}")
                    )
                if feedback is not None:
                    ts = datetime.fromtimestamp(feedback.pts / 1e9).strftime("%H:%M:%S")
                    self._history.append(
                        MessageFrame.new(role="assistant", content=f"[{ts}] {feedback.text}")
                    )
                if vision is not None:
                    ts = datetime.fromtimestamp(vision.pts / 1e9).strftime("%H:%M:%S")
                    self._history.append(
                        MessageFrame.new(role="system", content=f"[{ts}] {vision.text}")
                    )
                if memory is not None:
                    ts = datetime.fromtimestamp(memory.pts / 1e9).strftime("%H:%M:%S")
                    self._history.append(
                        MessageFrame.new(role="system", content=f"[{ts}] {memory.text}")
                    )
                if tc is not None:
                    ts = datetime.fromtimestamp(tc.pts / 1e9).strftime("%H:%M:%S")
                    self._history.append(
                        MessageFrame.new(
                            role="assistant",
                            content=f"[{ts}]",
                            tool_calls=[tc],
                        )
                    )
                if tr is not None:
                    ts = datetime.fromtimestamp(tr.pts / 1e9).strftime("%H:%M:%S")
                    self._history.append(
                        MessageFrame.new(
                            role="tool",
                            content=f"[{ts}] {tr.content}",
                            tool_call_id=tr.call_id,
                        )
                    )

            # Diff objects against previous state
            if objects_it is not None:
                obj_frame = next(objects_it)
                if obj_frame is not None:
                    self._diff_objects(obj_frame)

            # Build final messages: history
            msgs = self._history.copy()

            # visible objects
            visible_msg = self._build_visible_message()
            if visible_msg is not None:
                msgs.append(visible_msg)

            # Read agent spatial state (position + direction)
            if pose_it is not None:
                pose_frame = next(pose_it)
                if pose_frame is not None:
                    poses = pose_frame.get()
                    waist = poses.get("waist")
                    if waist is not None:
                        px, py, pz = waist.pos_x, waist.pos_y, waist.pos_z
                        heading = self._heading_from_quat(waist.rot_w, waist.rot_x, waist.rot_y, waist.rot_z)
                        msgs.append(
                            MessageFrame.new(
                                role="system",
                                content=f"[Position (y-up): x={px:.2f}, y={py:.2f}, z={pz:.2f}]\n"
                                        f"[Heading (from +Z clockwise): {heading:.0f}°]",
                            )
                        )

            self._print_messages(msgs)
            outputs.messages.send(msgs)

        print("[AgentState] Agent State management stopped")
