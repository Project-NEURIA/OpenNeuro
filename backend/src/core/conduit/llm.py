from __future__ import annotations

from typing import Any, NamedTuple

import litellm
from litellm import completion
from litellm.types.utils import ModelResponseStream
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import (
    EOS,
    InterruptFrame,
    MessageFrame,
    TextFrame,
    ToolCall,
    ToolDef,
)

litellm.drop_params = True


class LLMConfig(BaseModel):
    model: str = "groq/llama-3.3-70b-versatile"
    top_p: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class LLMInputs(NamedTuple):
    messages: Receiver[list[MessageFrame]]
    tools: Receiver[ToolDef] | None = None
    interrupt: Receiver[InterruptFrame] | None = None


class LLMOutputs(NamedTuple):
    token: Sender[TextFrame | EOS]
    text: Sender[TextFrame] | None = None
    tool_calls: Sender[ToolCall] | None = None
    eos: Sender[EOS] | None = None


class LLM(ThreadedComponent[LLMInputs, LLMOutputs]):
    description = "Generates text responses using a **large language model** via *litellm*. Accepts `MessageFrame` conversations and optional `ToolDef` definitions, streaming `TextFrame` tokens with support for *tool calls* and *interrupts*."

    """LLM text generation component using litellm."""

    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config

    def setup(self) -> None:
        """Warm up litellm's tokenizer so the first real call doesn't pay the
        ~300ms from_pretrained cost."""
        try:
            completion(
                model=self.config.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
            )
        except Exception:
            pass  # Warmup failure is non-fatal

    def run(self, inputs: LLMInputs, outputs: LLMOutputs) -> None:
        print("[LLM] Starting LLM generation")

        if inputs.interrupt is not None:
            inputs.interrupt.blocking = False

        # Collect tool definitions (drain once, tools are registered at start)
        tool_defs: list[dict[str, Any]] = []
        if inputs.tools is not None:
            inputs.tools.blocking = False
            for td in inputs.tools:
                if td is None:
                    break
                tool_defs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": td.name,
                            "description": td.description,
                            "parameters": td.parameters,
                            **({"strict": td.strict} if td.strict is not None else {}),
                        },
                    }
                )
        print(
            f"[LLM] Tools: {[td['function']['name'] for td in tool_defs]}"
            if tool_defs
            else "[LLM] No tools"
        )

        for messages in inputs.messages:
            if messages is None:
                break

            msgs: list[Any] = []
            for m in messages:
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
                msgs.append(msg)
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": msgs,
                "stream": True,
                **(
                    {"top_p": self.config.top_p}
                    if self.config.top_p is not None
                    else {}
                ),
                **(
                    {"temperature": self.config.temperature}
                    if self.config.temperature is not None
                    else {}
                ),
                **(
                    {"max_tokens": self.config.max_tokens}
                    if self.config.max_tokens is not None
                    else {}
                ),
            }
            if tool_defs:
                kwargs["tools"] = tool_defs
                kwargs["tool_choice"] = "auto"

            stream = completion(**kwargs)

            tool_call_acc: dict[int, dict[str, str]] = {}
            full_text = ""

            for chunk in stream:
                if not isinstance(chunk, ModelResponseStream):
                    continue

                if inputs.interrupt is not None:
                    irq = next(inputs.interrupt)
                    if irq is not None:
                        print(f"[LLM] Interrupt: {irq.reason}")
                        if full_text and outputs.text is not None:
                            outputs.text.send(TextFrame.new(text=full_text))
                        outputs.token.send(EOS.END)
                        if outputs.eos is not None:
                            outputs.eos.send(EOS.END)
                        break

                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = getattr(choice, "delta", None)
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        acc = tool_call_acc.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            acc["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] += tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    if tool_call_acc and outputs.tool_calls is not None:
                        for acc in tool_call_acc.values():
                            outputs.tool_calls.send(
                                ToolCall.new(
                                    call_id=acc["id"],
                                    name=acc["name"],
                                    arguments=acc["arguments"],
                                )
                            )
                    if full_text and outputs.text is not None:
                        outputs.text.send(TextFrame.new(text=full_text))
                    outputs.token.send(EOS.END)
                    if outputs.eos is not None:
                        outputs.eos.send(EOS.END)
                    break

                text = (delta.content or "") if delta else ""
                if text:
                    full_text += text
                    outputs.token.send(TextFrame.new(text=text))

            print(f"[LLM] Generation done, text length: {len(full_text)}")

        print("[LLM] LLM generation stopped")
