from __future__ import annotations

from typing import TypedDict

import litellm

from src.core.component import Component
from src.core.channel import Channel


class LLMOutputs(TypedDict):
    text: Channel[str]


class LLM(Component[[Channel[str]], LLMOutputs]):
    def __init__(self) -> None:
        super().__init__()
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant. Keep your responses short and conversational."},
        ]
        self._output_text: Channel[str] = Channel(name="text")

    def output_channels(self) -> LLMOutputs:
        return {"text": self._output_text}

    def run(self, text: Channel[str]) -> None:
        for chunk in text.stream(self.stop_event):
            if chunk is None:
                break
            self._messages.append({"role": "user", "content": chunk})

            response = litellm.completion(
                model="groq/llama-3.3-70b-versatile",
                messages=self._messages,
                stream=True,
                top_p=0.97,
                temperature=1.08,
                max_tokens=350,
                stop=["\n"],
            )

            assistant_text = ""
            for resp_chunk in response:
                token = resp_chunk.choices[0].delta.content or ""
                if token:
                    assistant_text += token
                    self._output_text.send(token)

            self._messages.append({"role": "assistant", "content": assistant_text})
            self._output_text.send("")  # done sentinel
