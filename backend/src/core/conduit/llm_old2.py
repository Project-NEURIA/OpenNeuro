# from __future__ import annotations
#
# import os
# from typing import Any, NamedTuple
#
# from openai import OpenAI, Stream
# from openai.types.chat import ChatCompletionChunk
# from pydantic import BaseModel
#
# from src.core.channel import Receiver, Sender
# from src.core.component import ThreadedComponent, Tag
# from src.core.frames import EOS, InterruptFrame, MessageFrame, TextFrame, ToolCall, ToolDef
#
#
# class LLMConfig(BaseModel):
#     base_url: str = "https://api.groq.com/openai/v1"
#     model_id: str = "llama-3.3-70b-versatile"
#     api_key_env_var: str = "GROQ_API_KEY"
#     top_p: float = 0.97
#     temperature: float = 1.08
#     max_tokens: int = 350
#
#
# class LLMInputs(NamedTuple):
#     messages: Receiver[list[MessageFrame]]
#     tools: Receiver[ToolDef] | None = None
#     interrupt: Receiver[InterruptFrame] | None = None
#
#
# class LLMOutputs(NamedTuple):
#     token: Sender[TextFrame | EOS]
#     text: Sender[TextFrame] | None = None
#     tool_calls: Sender[ToolCall] | None = None
#
#
# class LLM(ThreadedComponent[LLMInputs, LLMOutputs]):
#     description = "Generates text responses using a large language model"
#
#     """LLM text generation component using OpenAI-compatible API."""
#
#     tags = Tag(io={"conduit"}, functionality={"llm"})
#
#     def __init__(self, config: LLMConfig) -> None:
#         super().__init__()
#         self.config = config
#         self._client = OpenAI(
#             base_url=config.base_url,
#             api_key=os.getenv(config.api_key_env_var) or "",
#         )
#
#     def run(self, inputs: LLMInputs, outputs: LLMOutputs) -> None:
#         print("[LLM] Starting LLM generation")
#
#         interrupt_it = (
#             inputs.interrupt(self, no_block=True)
#             if inputs.interrupt is not None
#             else None
#         )
#
#         # Collect tool definitions (drain once, tools are registered at start)
#         tool_defs: list[dict[str, Any]] = []
#         if inputs.tools is not None:
#             for td in inputs.tools(self, no_block=True, latest=False):
#                 if td is None:
#                     break
#                 tool_defs.append({
#                     "type": "function",
#                     "function": {
#                         "name": td.name,
#                         "description": td.description,
#                         "parameters": td.parameters,
#                         **({"strict": td.strict} if td.strict is not None else {}),
#                     },
#                 })
#         print(f"[LLM] Tools: {[td['function']['name'] for td in tool_defs]}" if tool_defs else "[LLM] No tools")
#         for messages in inputs.messages(self):
#             if messages is None:
#                 break
#
#             msgs: list[Any] = [
#                 {"role": m.role, "content": m.content} for m in messages
#             ]
#             kwargs: dict[str, Any] = {
#                 "model": self.config.model_id,
#                 "messages": msgs,
#                 "stream": True,
#                 "top_p": self.config.top_p,
#                 "temperature": self.config.temperature,
#                 "max_tokens": self.config.max_tokens,
#             }
#             if tool_defs:
#                 kwargs["tools"] = tool_defs
#             stream: Stream[ChatCompletionChunk] = self._client.chat.completions.create(**kwargs)
#
#             tool_call_acc: dict[int, dict[str, str]] = {}
#             full_text = ""
#
#             for chunk in stream:
#                 if interrupt_it is not None:
#                     irq = next(interrupt_it)
#                     if irq is not None:
#                         print(f"[LLM] Interrupt: {irq.reason}")
#                         if full_text and outputs.text is not None:
#                             outputs.text.send(TextFrame.new(text=full_text))
#                         outputs.token.send(EOS.END)
#                         break
#
#                 choice = chunk.choices[0] if chunk.choices else None
#                 if choice is None:
#                     continue
#
#                 delta = choice.delta
#                 if delta and delta.tool_calls:
#                     for tc in delta.tool_calls:
#                         acc = tool_call_acc.setdefault(
#                             tc.index, {"id": "", "name": "", "arguments": ""}
#                         )
#                         if tc.id:
#                             acc["id"] += tc.id
#                         if tc.function:
#                             if tc.function.name:
#                                 acc["name"] += tc.function.name
#                             if tc.function.arguments:
#                                 acc["arguments"] += tc.function.arguments
#
#                 if choice.finish_reason:
#                     if tool_call_acc and outputs.tool_calls is not None:
#                         for acc in tool_call_acc.values():
#                             outputs.tool_calls.send(
#                                 ToolCall.new(
#                                     call_id=acc["id"],
#                                     name=acc["name"],
#                                     arguments=acc["arguments"],
#                                 )
#                             )
#                     if full_text and outputs.text is not None:
#                         outputs.text.send(TextFrame.new(text=full_text))
#                     outputs.token.send(EOS.END)
#                     break
#
#                 text = delta.content or "" if delta else ""
#                 if text:
#                     full_text += text
#                     outputs.token.send(TextFrame.new(text=text))
