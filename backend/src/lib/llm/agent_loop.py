from __future__ import annotations

import math
from datetime import datetime
from typing import Any, NamedTuple

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
from src.core.config import PROJECTS_DIR, AppConfig
from src.core.utils import drain


class AgentLoopConfig(BaseModel):
    system_prompt: str
    post_prompt: str = (
        "Do nothing unless the user has talked and you haven't replied, "
        "or if you are executing a task."
    )


class AgentLoopInputs(NamedTuple):
    initial_msgs: Receiver[list[MessageFrame]] | None = None
    speech: Receiver[TextFrame] | None = None
    feedback: Receiver[TextFrame] | None = None
    tool_call: Receiver[ToolCall] | None = None
    tool_result: Receiver[ToolResult] | None = None
    vision: Receiver[TextFrame] | None = None
    pose: Receiver[BodyPoseFrame] | None = None
    objects: Receiver[ObjectLocationFrame] | None = None
    memory: Receiver[TextFrame] | None = None
    pause: Receiver[TextFrame] | None = None


class AgentLoopOutputs(NamedTuple):
    messages: Sender[list[MessageFrame]]


class AgentLoop(ThreadedComponent[AgentLoopInputs, AgentLoopOutputs]):
    """Manages conversation history, optionally enriched by memory and character card."""

    description = "Tracks and manages **agent conversation state**. Maintains message history enriched by optional *memory* and *character card* inputs, and outputs assembled `MessageFrame` lists for the LLM."
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: AgentLoopConfig) -> None:
        super().__init__()
        self.config = config
        self._history: list[MessageFrame] = [
            MessageFrame.new(role="system", content=config.system_prompt)
        ]

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

    def run(self, inputs: AgentLoopInputs, outputs: AgentLoopOutputs) -> None:
        print("[AgentLoop] Starting Agent Loop")

        # Read initial prompts once (constant component, e.g. CharacterCard)
        if inputs.initial_msgs is not None:
            frame = next(inputs.initial_msgs)
            if frame is not None:
                self._history = frame + self._history
                print(f"[AgentLoop] Initial messages loaded ({len(frame)} msgs)")

        # Configure receiver modes for optional inputs
        if inputs.speech:
            inputs.speech.blocking = False
        if inputs.feedback:
            inputs.feedback.blocking = False
        if inputs.vision:
            inputs.vision.newest = True
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
        if inputs.pause is not None:
            inputs.pause.newest = True
            inputs.pause.blocking = False

        # Buffer tool_calls until their matching tool_result arrives
        pending_tool_calls: dict[str, ToolCall] = {}

        while not self.stop_event.is_set():
            # Check for pause signal
            if inputs.pause is not None:
                p = next(inputs.pause, None)
                if p is not None:
                    print("[AgentLoop] Paused — draining requests")
                    continue

            # Drain everything except tool results and vision
            for speech, feedback, memory, tc in drain(
                inputs.speech,
                inputs.feedback,
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
                if memory is not None:
                    ts = datetime.fromtimestamp(memory.pts / 1e9).strftime("%H:%M:%S")
                    msg = MessageFrame.new(
                        role="system", content=f"[{ts}] {memory.text}"
                    )
                    self._history.append(msg)
                    self._print_message(msg)
                if tc is not None:
                    # Tool call in chronological position + placeholder result
                    ts = datetime.fromtimestamp(tc.pts / 1e9).strftime("%H:%M:%S")
                    msg_tc = MessageFrame.new(
                        role="assistant",
                        content="",
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
                                content=tr.content,
                                tool_call_id=tr.call_id,
                            )
                            self._print_message(self._history[i])
                            pending_tool_calls.pop(tr.call_id, None)
                            break

            # Build final messages: history
            msgs = self._history.copy()

            # Latest vision caption (transient, not in history)
            if inputs.vision is not None:
                vision_frame = next(inputs.vision, None)
                if vision_frame is not None:
                    ts = datetime.fromtimestamp(vision_frame.pts / 1e9).strftime(
                        "%H:%M:%S"
                    )
                    msgs.append(
                        MessageFrame.new(
                            role="system",
                            content=f"[{ts}] {vision_frame.text}",
                        )
                    )

            # Latest object locations (transient, not in history)
            if inputs.objects is not None:
                obj_frame = next(inputs.objects, None)
                if obj_frame is not None and len(obj_frame.labels) > 0:
                    lines = []
                    for i in range(len(obj_frame.labels)):
                        x, y, z = obj_frame.positions[i]
                        lines.append(
                            f'  "{obj_frame.labels[i]}" at ({x:.2f}, {y:.2f}, {z:.2f})'
                        )
                    msgs.append(
                        MessageFrame.new(
                            role="system",
                            content="[Currently visible objects]\n" + "\n".join(lines),
                        )
                    )

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

            # Post-prompt (always last)
            if self.config.post_prompt:
                msgs.append(
                    MessageFrame.new(role="system", content=self.config.post_prompt)
                )

            self._dump_messages(msgs)
            outputs.messages.send(msgs)

        print("[AgentLoop] Agent Loop stopped")

    @staticmethod
    def _dump_messages(msgs: list[MessageFrame]) -> None:
        """Serialize messages (as the LLM sees them) to the current project dir."""
        import json

        config = AppConfig.load_config()
        if not config.current_project:
            return
        path = PROJECTS_DIR / config.current_project / "messages.json"
        serialized = []
        for m in msgs:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            serialized.append(msg)
        path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False))
