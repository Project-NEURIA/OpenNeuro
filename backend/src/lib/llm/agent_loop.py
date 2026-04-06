from __future__ import annotations

import base64
import io
import json
import math
from datetime import datetime
from typing import Any, NamedTuple

from PIL import Image

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
    ToolResult,
    VideoFrame,
    VideoDataFormat,
)
from src.core.utils import drain


_DEFAULT_SYSTEM_PROMPT = """\
You are an embodied AI agent operating inside a real-time multimodal pipeline.

You exist in a physical or virtual environment. You can see through a camera, \
hear through a microphone, and act through tools (speaking, moving, etc.).

# Vision
Each turn, you may receive an image — this is your current view of the world. \
Use it to ground your responses in what you actually see. \
Do not hallucinate objects or spatial relationships that are not visible.

# Speech
User speech arrives as transcribed text with timestamps and speaker diarization. There could be more than one speaker.\
You should use your vision to help you ground who actually said the latest message from the user. \
Respond naturally and conversationally. Keep responses brief unless asked to elaborate.

# Tools
You have access to tools for interacting with the environment. \
Use them when the user asks you to act, not just describe. \
When you call a tool, wait for the result before continuing.

# Guidelines
- Be proactive. Do things when you are along. Think constantly but only talk when the latest user message hasn't been responded to.\
- If you see something relevant to the conversation, mention it naturally.
- Do not repeat yourself or narrate your own actions unless asked.
"""

_DEFAULT_POST_PROMPT = "Do nothing unless the user has talked and you haven't replied, "


class AgentLoopConfig(BaseModel):
    model: str = "gpt-4.1"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    post_prompt: str = _DEFAULT_POST_PROMPT


class AgentLoopInputs(NamedTuple):
    initial_msgs: Receiver[list[MessageFrame]] | None = None
    speech: Receiver[TextFrame] | None = None
    tool_result: Receiver[ToolResult] | None = None
    video: Receiver[VideoFrame] | None = None
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

    Uses the OpenAI Responses API with streaming. Drains speech and other
    inputs each iteration, builds the message context, calls the model,
    and streams tokens + tool calls to outputs.
    """

    description = (
        "Self-contained **agentic loop** with built-in LLM. "
        "Manages conversation history, drains multimodal inputs, "
        "and streams tokens and tool calls using the OpenAI Responses API."
    )
    tags = Tag(io={"conduit"}, functionality={"llm"})

    _DIARY_TOOL: dict[str, Any] = {
        "type": "function",
        "name": "diary",
        "description": (
            "Write a diary entry to track your mental state. "
            "Record what you see, where you are, what you're doing, "
            "how you feel, your current goal, and any observations. "
            "Call this every few turns to maintain self-awareness. "
            "Your diary entries stay in your history so you can always "
            "look back at what you were thinking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "A natural language snapshot of your current mental state.",
                },
            },
            "required": ["entry"],
        },
    }

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

    @staticmethod
    def _encode_frame(frame: VideoFrame) -> str:
        """Encode a VideoFrame as a base64 JPEG data URL."""
        img = Image.fromarray(frame.get(VideoDataFormat.RGB))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    # -- Main loop --

    def run(self, inputs: AgentLoopInputs, outputs: AgentLoopOutputs) -> None:
        print("[AgentLoop] Starting")

        self._input = []

        # Read initial prompts once (e.g. CharacterCard)
        if inputs.initial_msgs is not None:
            frame = next(inputs.initial_msgs)
            if frame is not None:
                for m in frame:
                    self._append_msg(m.role, m.content or "")

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
        # Always include the built-in diary tool
        self._tool_defs.append(self._DIARY_TOOL)
        print(f"[AgentLoop] Tools: {[t['name'] for t in self._tool_defs]}")

        # Configure receiver modes
        if inputs.speech:
            inputs.speech.blocking = False
        if inputs.memory:
            inputs.memory.blocking = False
        if inputs.video is not None:
            inputs.video.newest = True
            inputs.video.blocking = False
        if inputs.objects is not None:
            inputs.objects.newest = True
            inputs.objects.blocking = False
        if inputs.pose is not None:
            inputs.pose.newest = True
            inputs.pose.blocking = False

        while not self.stop_event.is_set():
            # Drain any new speech into history
            for (speech,) in drain(inputs.speech):
                if speech is not None:
                    ts = datetime.fromtimestamp(speech.pts / 1e9).strftime("%H:%M:%S")
                    self._append_msg("user", f"[{ts}] {speech.text}")

            # Build transient context (not persisted in history)
            transient: list[dict[str, Any]] = []

            # Latest video frame — what the agent currently sees
            if inputs.video is not None:
                vf = next(inputs.video, None)
                if vf is not None:
                    transient.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": self._encode_frame(vf),
                                },
                                {
                                    "type": "input_text",
                                    "text": "[This is what you currently see]",
                                },
                            ],
                        }
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
                    head = poses.get("head")
                    print(f"[AgentLoop] waist={waist} head={head}")
                    if waist is not None:
                        body_heading = self._heading_from_quat(
                            waist.rot_w, waist.rot_x, waist.rot_y, waist.rot_z
                        )
                        lines = [
                            f"[Position (y-up): x={waist.pos_x:.2f}, y={waist.pos_y:.2f}, z={waist.pos_z:.2f}]",
                            f"[Body heading (from +Z clockwise): {body_heading:.0f}°]",
                        ]
                        if head is not None:
                            look_heading = self._heading_from_quat(
                                head.rot_w, head.rot_x, head.rot_y, head.rot_z
                            )
                            lines.append(f"[Look heading (from +Z clockwise): {look_heading:.0f}°]")
                        pose_content = "\n".join(lines)
                        print(f"[AgentLoop] Pose context: {pose_content}")
                        transient.append(
                            {
                                "role": "system",
                                "content": pose_content,
                            }
                        )

            if self.config.post_prompt:
                transient.append({"role": "system", "content": self.config.post_prompt})

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
            "instructions": self.config.system_prompt,
            "input": api_input,
            "stream": True,
            "reasoning": {"effort": "low"},
        }
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
                    tool_calls.append({"call_id": "", "name": "", "arguments": ""})
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

        # Process tool calls
        for tc in tool_calls:
            self._input.append(
                {
                    "type": "function_call",
                    "call_id": tc["call_id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
            )

            if tc["name"] == "diary":
                # Built-in: handle internally
                try:
                    entry = json.loads(tc["arguments"]).get("entry", "")
                except (json.JSONDecodeError, KeyError):
                    entry = tc["arguments"]
                print(f"[AgentLoop] Diary: {entry[:120]}")
                self._input.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc["call_id"],
                        "output": "ok",
                    }
                )
            else:
                # External tool: emit and wait for result
                if outputs.tool_calls is not None:
                    outputs.tool_calls.send(
                        ToolCall.new(
                            call_id=tc["call_id"],
                            name=tc["name"],
                            arguments=tc["arguments"],
                        )
                    )
                if inputs.tool_result is not None:
                    result = next(inputs.tool_result, None)
                    if result is not None:
                        self._input.append(
                            {
                                "type": "function_call_output",
                                "call_id": result.call_id,
                                "output": result.content,
                            }
                        )

        # EOS
        outputs.token.send(EOS.END)
        if outputs.eos is not None:
            outputs.eos.send(EOS.END)
