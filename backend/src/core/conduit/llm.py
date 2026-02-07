from __future__ import annotations

import litellm

from ..node import Node
from ..topic import Topic, Stream


class LLM(Node[str]):
    def __init__(self, input_stream: Stream[str]) -> None:
        self._input_stream = input_stream
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant. Keep your responses short and conversational."},
        ]
        super().__init__(Topic[str]())

    def run(self) -> None:
        for text in self._input_stream:
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
                    self.topic.send(token)

            self._messages.append({"role": "assistant", "content": assistant_text})
            self.topic.send("")  # done sentinel
