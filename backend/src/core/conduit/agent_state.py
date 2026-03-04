from __future__ import annotations

import threading
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import InterruptFrame, MessagesFrame, TextFrame


class AgentStateConfig(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."
    user_name: str = "User"
    chatbot_name: str = "Assistant"


class AgentStateInputs(NamedTuple):
    asr: Receiver[TextFrame]
    feedback: Receiver[TextFrame]
    interrupt: Receiver[InterruptFrame] | None = None
    memory_prefix: Receiver[TextFrame] | None = None


class AgentStateOutputs(NamedTuple):
    messages: Sender[MessagesFrame]
    messages_for_memory: Sender[MessagesFrame] | None = None


class AgentState(Component[AgentStateInputs, AgentStateOutputs]):
    """Manages conversation history, optionally enriched by Mem0 memory retrieval."""

    def __init__(self, config: AgentStateConfig) -> None:
        super().__init__()
        self.config = config
        self._history: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def _build_context(self, memory_prefix: str = "") -> str:
        with self._lock:
            lines = [self.config.system_prompt]
            if memory_prefix:
                lines += ["", memory_prefix]
            lines.append("***")
            lines += [f"{n}: {t}" for n, t in self._history]
            lines.append(f"{self.config.chatbot_name}:")
            return "\n".join(lines)

    def _build_messages(self, memory_prefix: str = "") -> list[dict[str, str]]:
        with self._lock:
            system = self.config.system_prompt
            if memory_prefix:
                system = f"{system}\n\n{memory_prefix}"
            messages = [{"role": "system", "content": system}]
            for name, text in self._history:
                role = "user" if name == self.config.user_name else "assistant"
                messages.append({"role": role, "content": text})
            return messages

    def run(self, inputs: AgentStateInputs, outputs: AgentStateOutputs) -> None:
        print("[AgentState] Starting Agent State management")

        has_memory = inputs.memory_prefix is not None and outputs.messages_for_memory is not None
        memory_gen = inputs.memory_prefix(self) if has_memory else None
        print(f"[AgentState] Memory integration {'enabled' if has_memory else 'disabled'}")

        def process_asr() -> None:
            for text_frame in inputs.asr(self):
                if text_frame is None:
                    break

                text = text_frame.text.strip()
                if not text:
                    continue

                with self._lock:
                    self._history.append((self.config.user_name, text))
                print(f"[AgentState] User: {text}")

                # Memory retrieval (optional)
                mem_text = ""
                if has_memory:
                    outputs.messages_for_memory.send(
                        MessagesFrame.new(
                            text=self._build_context(),
                            messages=self._build_messages(),
                        )
                    )
                    # Block until Mem0 returns the memory prefix
                    prefix_frame = next(memory_gen)
                    if prefix_frame is not None and prefix_frame.text:
                        mem_text = prefix_frame.text
                        print(f"[AgentState] Memory prefix injected {mem_text}")

                outputs.messages.send(
                    MessagesFrame.new(
                        text=self._build_context(memory_prefix=mem_text),
                        messages=self._build_messages(memory_prefix=mem_text),
                    )
                )

        def process_feedback() -> None:
            for text_frame in inputs.feedback(self):
                if text_frame is None:
                    break

                chunk = text_frame.text
                if not chunk:
                    continue

                with self._lock:
                    # Append or start new assistant message
                    if (
                        self._history
                        and self._history[-1][0] == self.config.chatbot_name
                    ):
                        name, text = self._history[-1]
                        self._history[-1] = (name, text + chunk)
                    else:
                        self._history.append((self.config.chatbot_name, chunk))

        def process_interrupts() -> None:
            if inputs.interrupt is None:
                return
            for frame in inputs.interrupt(self):
                if frame is None:
                    break
                print(f"[AgentState] Interrupt received: {frame.reason}")

        threads = [
            threading.Thread(target=process_asr, daemon=True),
            threading.Thread(target=process_feedback, daemon=True),
            threading.Thread(target=process_interrupts, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        print("[AgentState] Agent State management stopped")
