from __future__ import annotations

import threading
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent
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
    observation: Receiver[TextFrame] | None = None


class AgentStateOutputs(NamedTuple):
    messages: Sender[MessagesFrame]
    messages_for_memory: Sender[MessagesFrame] | None = None


class AgentState(PrimitiveComponent[AgentStateInputs, AgentStateOutputs]):
    """Manages conversation history, optionally enriched by Mem0 memory retrieval."""

    def __init__(self, config: AgentStateConfig) -> None:
        super().__init__()
        self.config = config
        self._history: list[tuple[str, str]] = []
        self._latest_observation: str = ""
        self._lock = threading.Lock()

    def _build_context(self, memory_prefix: str = "", observation: str = "") -> str:
        with self._lock:
            lines = [self.config.system_prompt]
            if memory_prefix:
                lines += ["", memory_prefix]
            if observation:
                lines += ["", f"[Current Vision Observation]\n{observation}"]
            lines.append("***")
            lines += [f"{n}: {t}" for n, t in self._history]
            lines.append(f"{self.config.chatbot_name}:")
            return "\n".join(lines)

    def _build_messages(
        self, memory_prefix: str = "", observation: str = ""
    ) -> list[dict[str, str]]:
        with self._lock:
            system = self.config.system_prompt
            if memory_prefix or observation:
                parts = [system]
                if memory_prefix:
                    parts.append(memory_prefix)
                if observation:
                    parts.append(f"[Current Vision Observation]\n{observation}")
                system = "\n\n".join(parts)
            messages = [{"role": "system", "content": system}]
            for name, text in self._history:
                role = "user" if name == self.config.user_name else "assistant"
                messages.append({"role": role, "content": text})
            return messages

    def run(self, inputs: AgentStateInputs, outputs: AgentStateOutputs) -> None:
        print("[AgentState] Starting Agent State management")

        memory_prefix_receiver = inputs.memory_prefix
        memory_sender = outputs.messages_for_memory
        has_memory = memory_prefix_receiver is not None and memory_sender is not None
        memory_gen = (
            memory_prefix_receiver(self) if memory_prefix_receiver is not None else None
        )
        print(
            f"[AgentState] Memory integration {'enabled' if has_memory else 'disabled'}"
        )

        def process_observation() -> None:
            if inputs.observation is None:
                return
            for obs_frame in inputs.observation(self):
                if obs_frame is None:
                    break
                with self._lock:
                    self._latest_observation = obs_frame.text
                print(f"[AgentState] Observation updated: {obs_frame.text}")

        def process_asr() -> None:
            for text_frame in inputs.asr(self):
                if text_frame is None:
                    break

                text = text_frame.text.strip()
                if not text:
                    continue

                with self._lock:
                    self._history.append((self.config.user_name, text))
                    current_obs = self._latest_observation
                print(f"[AgentState] User: {text}")

                # Memory retrieval (optional)
                mem_text = ""
                if has_memory and memory_sender is not None and memory_gen is not None:
                    memory_sender.send(
                        MessagesFrame.new(
                            text=self._build_context(observation=current_obs),
                            messages=self._build_messages(observation=current_obs),
                        )
                    )
                    # Block until Mem0 returns the memory prefix
                    prefix_frame = next(memory_gen)
                    if prefix_frame is not None and prefix_frame.text:
                        mem_text = prefix_frame.text
                        print(f"[AgentState] Memory prefix injected {mem_text}")

                outputs.messages.send(
                    MessagesFrame.new(
                        text=self._build_context(
                            memory_prefix=mem_text, observation=current_obs
                        ),
                        messages=self._build_messages(
                            memory_prefix=mem_text, observation=current_obs
                        ),
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
            threading.Thread(target=process_observation, daemon=True),
            threading.Thread(target=process_asr, daemon=True),
            threading.Thread(target=process_feedback, daemon=True),
            threading.Thread(target=process_interrupts, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        print("[AgentState] Agent State management stopped")
