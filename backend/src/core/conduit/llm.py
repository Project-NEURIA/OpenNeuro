from __future__ import annotations

import litellm

from ..node import Node
from ..topic import Topic, Stream


class LLM(Node[str]):
    def __init__(
        self,
        source: Stream[str],
        *,
        model: str = "groq/llama-3.3-70b-versatile",
        system_prompt: str = "",
        top_p: float = 0.97,
        temperature: float = 1.08,
        max_tokens: int = 350,
    ) -> None:
        self._source = source
        self._model = model
        self._top_p = top_p
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._messages: list[dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        self.output = Topic[str]()
        super().__init__(self.output)

    def run(self) -> None:
        for text in self._source:
            self._messages.append({"role": "user", "content": text})

            response = litellm.completion(
                model=self._model,
                messages=self._messages,
                stream=True,
                top_p=self._top_p,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=["\n"],
            )

            assistant_text = ""
            for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    assistant_text += token
                    self.output.send(token)

            self._messages.append({"role": "assistant", "content": assistant_text})
            self.output.send("")  # done sentinel
