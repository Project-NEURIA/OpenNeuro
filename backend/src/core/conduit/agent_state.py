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


class AgentStateOutputs(NamedTuple):
    messages: Sender[MessagesFrame]
    interrupt: Sender[InterruptFrame]


class AgentState(Component[AgentStateInputs, AgentStateOutputs]):
    """Agent state component that manages conversation history."""

    def __init__(self, config: AgentStateConfig) -> None:
        super().__init__()
        self.config: AgentStateConfig = config

        # Conversation history as list of (speaker, text) tuples
        self._history: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def _build_context(self) -> str:
        """Build single prompt string."""
        with self._lock:
            lines = [self.config.system_prompt, "***"]
            for name, text in self._history:
                lines.append(f"{name}: {text}")
            lines.append(f"{self.config.chatbot_name}:")
            return "\n".join(lines)

    def _build_messages(self) -> list[dict[str, str]]:
        """Build message list for Chat APIs."""
        with self._lock:
            messages = [{"role": "system", "content": self.config.system_prompt}]
            for name, text in self._history:
                role = "user" if name == self.config.user_name else "assistant"
                messages.append({"role": role, "content": text})
            return messages

    def run(self, inputs: AgentStateInputs, outputs: AgentStateOutputs) -> None:
        print("[AgentState] Starting Agent State management")

        def process_asr() -> None:
            for text_frame in inputs.asr(self):
                if text_frame is None:
                    break

                text = text_frame.get().strip()
                if not text:
                    continue

                with self._lock:
                    self._history.append((self.config.user_name, text))

                print(f"[AgentState] User: {text}")

                # Output context as MessagesFrame
                outputs.messages.send(
                    MessagesFrame(
                        display_name="agent_state",
                        text=self._build_context(),
                        messages=self._build_messages(),
                        pts=text_frame.pts,
                    )
                )

        def process_feedback() -> None:
            for text_frame in inputs.feedback(self):
                if text_frame is None:
                    break

                chunk = text_frame.get()
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
                print(f"[AgentState] Interrupt received: {frame.get()}")
                # Forward the interrupt
                outputs.interrupt.send(frame)

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
