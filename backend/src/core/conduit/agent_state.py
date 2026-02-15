from __future__ import annotations

import threading
from typing import TypedDict

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import MessagesFrame, InterruptFrame, TextFrame
from src.core.config import BaseConfig


class AgentStateConfig(BaseConfig):
    system_prompt: str = "You are a helpful AI assistant."
    user_name: str = "User"
    chatbot_name: str = "Assistant"


class AgentStateOutputs(TypedDict):
    messages: Channel[MessagesFrame]
    interrupt: Channel[InterruptFrame]


class AgentState(
    Component[
        [Channel[TextFrame], Channel[TextFrame], Channel[InterruptFrame]],
        AgentStateOutputs,
    ]
):
    """Agent state component that manages conversation history."""

    def __init__(self, config: AgentStateConfig | None = None) -> None:
        super().__init__(config or AgentStateConfig())
        self.config: AgentStateConfig

        self._output_messages = Channel[MessagesFrame](name="messages")
        self._output_interrupt = Channel[InterruptFrame](name="interrupt")

        # Conversation history as list of (speaker, text) tuples
        self._history: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def get_output_channels(self) -> AgentStateOutputs:
        return {"messages": self._output_messages, "interrupt": self._output_interrupt}

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

    def run(
        self,
        asr: Channel[TextFrame] | None = None,
        feedback: Channel[TextFrame] | None = None,
        interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        print("[AgentState] Starting Agent State management")

        def process_asr():
            if not asr:
                return
            for text_frame in asr.stream(self):
                if text_frame is None:
                    break

                text = text_frame.get().strip()
                if not text:
                    continue

                with self._lock:
                    self._history.append((self.config.user_name, text))

                print(f"[AgentState] User: {text}")

                # Output context as MessagesFrame
                self._output_messages.send(
                    MessagesFrame(
                        display_name="agent_state",
                        text=self._build_context(),
                        messages=self._build_messages(),
                        pts=text_frame.pts,
                    )
                )

        def process_feedback():
            if not feedback:
                return
            for text_frame in feedback.stream(self):
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

        def process_interrupts():
            if not interrupt:
                return
            for frame in interrupt.stream(self):
                if frame is None:
                    break
                print(f"[AgentState] Interrupt received: {frame.get()}")
                # Forward the interrupt
                self._output_interrupt.send(frame)

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
