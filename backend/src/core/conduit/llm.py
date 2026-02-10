from __future__ import annotations

from typing import Never

import litellm

from ..component import Component
from ..topic import NOTOPIC, Topic


class LLM(Component[str, str]):
    def __init__(self) -> None:
        super().__init__()
        self._output = Topic[str]()
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant. Keep your responses short and conversational."},
        ]

    def get_output_topics(self) -> tuple[Topic[str], Topic[Never], Topic[Never], Topic[Never]]:
        return (self._output, NOTOPIC, NOTOPIC, NOTOPIC)

    def set_input_topics(self, t1: Topic[str], t2: Topic[Never] = NOTOPIC, t3: Topic[Never] = NOTOPIC, t4: Topic[Never] = NOTOPIC) -> None:
        self._input_topic = t1

    def run(self) -> None:
        for text in self._input_topic.stream(self.stop_event):
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
