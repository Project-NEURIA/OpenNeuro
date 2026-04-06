from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Any, NamedTuple

from openai import OpenAI
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import (
    BodyPoseFrame,
    EOS,
    MessageFrame,
    ObjectLocationFrame,
    TextFrame,
    ToolCall,
    ToolDef,
)
from src.core.config import PROJECTS_DIR, AppConfig
from src.core.utils import drain


class AgentLoopConfig(BaseModel):
    model: str = "gpt-4.1"
    system_prompt: str = "You are a helpful assistant."
    post_prompt: str = (
        "Do nothing unless the user has talked and you haven't replied, "
        "or if you are executing a task."
    )
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class AgentLoopInputs(NamedTuple):
    initial_msgs: Receiver[list[MessageFrame]] | None = None
    speech: Receiver[TextFrame] | None = None
    feedback: Receiver[TextFrame] | None = None
    tool_result: Receiver[TextFrame] | None = None
    vision: Receiver[TextFrame] | None = None
    pose: Receiver[BodyPoseFrame] | None = None
    objects: Receiver[ObjectLocationFrame] | None = None
    memory: Receiver[TextFrame] | None = None
    tools: Receiver[ToolDef] | None = None


class AgentLoopOutputs(NamedTuple):
    token: Sender[TextFrame | EOS]
    text: Sender[TextFrame] | None = None
    tool_calls: Sender[ToolCall] | None = None
    eos: Sender[EOS] | None = None


class AgentLoop(ThreadedComponent[AgentLoopInputs, AgentLoopOutputs]):
    """Agentic loop: manages conversation state and calls the LLM directly.

    Uses the OpenAI Responses API with streaming. Drains speech, feedback,
    vision, and other inputs each iteration, builds the message context,
    calls the model, and streams tokens + tool calls to outputs.
    """

    description = (
        "Self-contained **agentic loop** with built-in LLM. "
        "Manages conversation history, drains multimodal inputs, "
        "and streams tokens and tool calls using the OpenAI Responses API."
    )
    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: AgentLoopConfig) -> None:
        super().__init__()
        self.config = config
        self._client = OpenAI()
        self._input: list[dict[str, Any]] = []
        self._tool_defs: list[dict[str, Any]] = []

    # -- Helpers --

    @staticmethod
    def _heading_from_quat(w: float, x: float, y: float, z: float) -> float:
        fwd_x = 2 * (x * z + w * y)
        fwd_z = 1 - 2 * (x * x + y * y)
        return -math.degrees(math.atan2(fwd_x, fwd_z))

    def _append_msg(self, role: str, content: str) -> None:
        self._input.append({"role": role, "content": content})
        preview = content[:120]
        print(f"  [{role}] {preview}")

    # -- Main loop --

    def run(self, inputs: AgentLoopInputs, outputs: AgentLoopOutputs) -> None:
        print("[AgentLoop] Starting")

        # System prompt
        self._input = [{"role": "system", "content": self.config.system_prompt}]

        # Read initial prompts once (e.g. CharacterCard)
        if inputs.initial_msgs is not None:
            frame = next(inputs.initial_msgs)
            if frame is not None:
                for m in frame:
                    self._append_msg(m.role, m.content)

        # Collect tool definitions once
        if inputs.tools is not None:
            inputs.tools.blocking = False
            for td in inputs.tools:
                if td is None:
                    break
                self._tool_defs.append(
                    {
                        "type": "function",
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters,
                        **({"strict": td.strict} if td.strict is not None else {}),
                    }
                )
        if self._tool_defs:
            print(f"[AgentLoop] Tools: {[t['name'] for t in self._tool_defs]}")

        # Configure receiver modes
        if inputs.speech:
            inputs.speech.blocking = False
        if inputs.feedback:
            inputs.feedback.blocking = False
        if inputs.vision:
            inputs.vision.newest = True
            inputs.vision.blocking = False
        if inputs.memory:
            inputs.memory.blocking = False
        if inputs.tool_result:
            inputs.tool_result.blocking = False
        if inputs.objects is not None:
            inputs.objects.newest = True
            inputs.objects.blocking = False
        if inputs.pose is not None:
            inputs.pose.newest = True
            inputs.pose.blocking = False

        while not self.stop_event.is_set():
            # Drain inputs into history
            has_new_input = False

            for speech, feedback, memory in drain(
                inputs.speech,
                inputs.feedback,
                inputs.memory,
            ):
                if speech is not None:
                    ts = datetime.fromtimestamp(speech.pts / 1e9).strftime("%H:%M:%S")
                    self._append_msg("user", f"[{ts}] {speech.text}")
                    has_new_input = True
                if feedback is not None:
                    ts = datetime.fromtimestamp(feedback.pts / 1e9).strftime("%H:%M:%S")
                    self._append_msg("assistant", f"[{ts}] {feedback.text}")
                if memory is not None:
                    ts = datetime.fromtimestamp(memory.pts / 1e9).strftime("%H:%M:%S")
                    self._append_msg("system", f"[{ts}] {memory.text}")

            # Nothing new — sleep briefly to avoid busy loop
            if not has_new_input:
                self.stop_event.wait(0.1)
                continue

            # Build transient context (not persisted in history)
            transient: list[dict[str, Any]] = []

            if inputs.vision is not None:
                v = next(inputs.vision, None)
                if v is not None:
                    ts = datetime.fromtimestamp(v.pts / 1e9).strftime("%H:%M:%S")
                    transient.append(
                        {"role": "system", "content": f"[{ts}] {v.text}"}
                    )

            if inputs.objects is not None:
                obj = next(inputs.objects, None)
                if obj is not None and len(obj.labels) > 0:
                    lines = [
                        f'  "{obj.labels[i]}" at ({obj.positions[i][0]:.2f}, {obj.positions[i][1]:.2f}, {obj.positions[i][2]:.2f})'
                        for i in range(len(obj.labels))
                    ]
                    transient.append(
                        {
                            "role": "system",
                            "content": "[Currently visible objects]\n"
                            + "\n".join(lines),
                        }
                    )

            if inputs.pose is not None:
                pose = next(inputs.pose, None)
                if pose is not None:
                    poses = pose.get()
                    waist = poses.get("waist")
                    if waist is not None:
                        heading = self._heading_from_quat(
                            waist.rot_w, waist.rot_x, waist.rot_y, waist.rot_z
                        )
                        transient.append(
                            {
                                "role": "system",
                                "content": f"[Position (y-up): x={waist.pos_x:.2f}, y={waist.pos_y:.2f}, z={waist.pos_z:.2f}]\n"
                                f"[Heading (from +Z clockwise): {heading:.0f}°]",
                            }
                        )

            if self.config.post_prompt:
                transient.append(
                    {"role": "system", "content": self.config.post_prompt}
                )

            # Call LLM
            api_input = self._input + transient
            self._call_llm(api_input, inputs, outputs)

        print("[AgentLoop] Stopped")

    def _call_llm(
        self,
        api_input: list[dict[str, Any]],
        inputs: AgentLoopInputs,
        outputs: AgentLoopOutputs,
    ) -> None:
        """Stream a single LLM response, emit tokens/tool calls, handle tool loop."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": api_input,
            "stream": True,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.max_tokens is not None:
            kwargs["max_output_tokens"] = self.config.max_tokens
        if self._tool_defs:
            kwargs["tools"] = self._tool_defs
            kwargs["tool_choice"] = "auto"

        stream = self._client.responses.create(**kwargs)

        full_text = ""
        tool_calls: list[dict[str, str]] = []

        for event in stream:
            if self.stop_event.is_set():
                break

            if event.type == "response.output_text.delta":
                full_text += event.delta
                outputs.token.send(TextFrame.new(text=event.delta))

            elif event.type == "response.function_call_arguments.delta":
                # Accumulate function call arguments
                if not tool_calls or tool_calls[-1].get("_done"):
                    tool_calls.append(
                        {"call_id": "", "name": "", "arguments": ""}
                    )
                tool_calls[-1]["arguments"] += event.delta

            elif event.type == "response.output_item.added":
                if hasattr(event, "item") and event.item.type == "function_call":
                    tool_calls.append(
                        {
                            "call_id": event.item.call_id,
                            "name": event.item.name,
                            "arguments": "",
                        }
                    )

            elif event.type == "response.output_item.done":
                if (
                    hasattr(event, "item")
                    and event.item.type == "function_call"
                    and tool_calls
                ):
                    tool_calls[-1]["call_id"] = event.item.call_id
                    tool_calls[-1]["name"] = event.item.name
                    tool_calls[-1]["arguments"] = event.item.arguments
                    tool_calls[-1]["_done"] = True  # type: ignore[assignment]

        # Emit completed text
        if full_text:
            self._append_msg("assistant", full_text)
            if outputs.text is not None:
                outputs.text.send(TextFrame.new(text=full_text))

        # Emit tool calls
        if tool_calls and outputs.tool_calls is not None:
            for tc in tool_calls:
                outputs.tool_calls.send(
                    ToolCall.new(
                        call_id=tc["call_id"],
                        name=tc["name"],
                        arguments=tc["arguments"],
                    )
                )
                # Add function_call to input for next turn
                self._input.append(
                    {
                        "type": "function_call",
                        "call_id": tc["call_id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                )

            # Wait for tool results and add them
            if inputs.tool_result is not None:
                inputs.tool_result.blocking = True
                for tc in tool_calls:
                    result = next(inputs.tool_result, None)
                    if result is None:
                        break
                    self._input.append(
                        {
                            "type": "function_call_output",
                            "call_id": tc["call_id"],
                            "output": result.text,
                        }
                    )
                inputs.tool_result.blocking = False

                # Loop back — call LLM again with tool results
                if not self.stop_event.is_set():
                    self._call_llm(self._input, inputs, outputs)
                    return

        # EOS
        outputs.token.send(EOS.END)
        if outputs.eos is not None:
            outputs.eos.send(EOS.END)
