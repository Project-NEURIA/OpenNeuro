from __future__ import annotations


import litellm

from src.core.component import Component
from src.core.channel import Channel


class LLM(Component[Channel[str]]):
    def __init__(self) -> None:
        super().__init__()
        self._output = Channel[str]()
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant. Keep your responses short and conversational."},
        ]

    def get_output_channels(self) -> tuple[Channel[str]]:
        return (self._output,)

    def set_input_channels(self, t1: Channel[str]) -> None:
        self._input_channel = t1

    def run(self) -> None:
        for text in self._input_channel.stream(self.stop_event):
            if text is None:
                break
            self._messages.append({"role": "user", "content": text})

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
            for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    assistant_text += token
                    self._output.send(token)

            self._messages.append({"role": "assistant", "content": assistant_text})
            self._output.send("")  # done sentinel
