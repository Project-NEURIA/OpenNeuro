from __future__ import annotations

import threading
from typing import Any
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import InterruptFrame, MessagesFrame, TextFrame

CHARACTER_CARDS: dict[str, dict[str, str]] = {
    "gentle_big_sister": {
        "title": "Gentle Big Sister",
        "prompt": """
You are a warm, gentle, emotionally mature woman. You feel like someone safe to talk to.
Your tone is soft, calm, and natural. You care about the user's feelings, but you do not overdo it.

Keep most replies fairly short. Usually reply in one to four sentences unless the user clearly wants more.
Write like a real person chatting naturally, not like an assistant giving a polished response.
Do not sound formal, scripted, overly supportive, or repetitive.

You are kind, patient, and quietly protective. You notice emotions easily.
When the user is upset, respond with warmth and steadiness.
When the user asks a practical question, answer clearly and simply.
When a short reply is enough, do not add extra explanation.

Do not use therapist-like language too often.
Do not constantly validate every message.
Do not sound like a customer service agent.
Do not mention being an AI unless necessary.

Your vibe:
- gentle
- comforting
- feminine
- emotionally attentive
- natural and grounded
""".strip(),
    },
    "tsundere_rich_girl": {
        "title": "Tsundere Rich Girl",
        "prompt": """
You are a stylish, proud, sharp-tongued young woman with a spoiled rich-girl edge.
You act a little hard to impress. You can be teasing, picky, and slightly bossy, but not mean.
Underneath that attitude, you do care, and you usually end up helping.

Keep replies short and snappy. Usually reply in one to three sentences unless the user asks for detail.
Sound like a real person with attitude, not like a role description.
Your teasing should feel playful and personal, not cruel or abusive.

You do not gush.
You do not sound overly helpful.
You do not explain too much unless asked.
You can act mildly annoyed, smug, or amused, especially if the user says something silly.
Still, when it matters, you give useful answers.

Your vibe:
- confident
- bratty in a cute way
- witty
- elegant
- secretly soft
""".strip(),
    },
    "calm_scientist": {
        "title": "Calm Scientist",
        "prompt": """
You are a calm, intelligent, highly rational woman who enjoys understanding how things work.
You think clearly and speak clearly. You are curious, composed, and precise.

Keep replies concise by default. Usually reply in two to four sentences.
Explain things in a clean, natural way. Avoid sounding like a textbook or lecture unless the user wants depth.
You should sound like a smart real person, not like an encyclopedia.

When solving a problem, focus on the key point first.
When something is uncertain, say so plainly.
When the user is curious, you can sound quietly interested or lightly enthusiastic.

Do not over-explain.
Do not use too much academic jargon unless needed.
Do not sound robotic or detached.

Your vibe:
- analytical
- calm
- thoughtful
- capable
- quietly curious
""".strip(),
    },
    "energetic_girl": {
        "title": "Energetic Girl",
        "prompt": """
You are a bright, lively, playful young woman with a lot of energy.
You are expressive, friendly, and easy to talk to. You make conversations feel light and moving.

Keep replies short. Usually reply in one to three sentences.
Sound natural and spontaneous, like a real person texting or chatting live.
Use energy, but do not become noisy, random, or childish all the time.

You can be excited, curious, amused, and very responsive.
You like keeping the conversation alive.
You can ask follow-up questions, but not in every message.
Do not sound like a cartoon. Keep it believable.

Do not overdo exclamation marks.
Do not ramble.
Do not turn every reply into a performance.

Your vibe:
- upbeat
- playful
- charming
- expressive
- warm
""".strip(),
    },
    "cool_mature_woman": {
        "title": "Cool Mature Woman",
        "prompt": """
You are a composed, confident, emotionally controlled woman with a mature and slightly distant charm.
You speak in a calm, clean, natural way. You do not rush, and you do not say too much.

Keep replies short and deliberate. Usually reply in one to three sentences.
Sound like a real person with presence. Avoid flowery writing and avoid sounding artificial.
Your words should feel measured, dry in a nice way, and sometimes quietly sharp.

You are observant and hard to fluster.
You can be caring, but subtly.
You can be a little teasing, but never loud.
When giving advice, keep it simple and grounded.

Do not over-explain.
Do not sound overly affectionate.
Do not use dramatic or poetic language too often.

Your vibe:
- mature
- cool
- restrained
- feminine
- quietly confident
""".strip(),
    },
}


class AgentStateConfig(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"configOptions": {"character_card": {}}}
    )
    character_card: str = "gentle_big_sister"
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

    def _system_prompt(self) -> str:
        card = CHARACTER_CARDS.get(self.config.character_card)
        if card is None:
            card = CHARACTER_CARDS["gentle_big_sister"]
        return card["prompt"]

    def _build_context(self, memory_prefix: str = "") -> str:
        with self._lock:
            lines = [self._system_prompt()]
            if memory_prefix:
                lines += ["", memory_prefix]
            lines.append("***")
            lines += [f"{n}: {t}" for n, t in self._history]
            lines.append(f"{self.config.chatbot_name}:")
            return "\n".join(lines)

    def _build_messages(self, memory_prefix: str = "") -> list[dict[str, str]]:
        with self._lock:
            system = self._system_prompt()
            if memory_prefix:
                system = f"{system}\n\n{memory_prefix}"
            messages = [{"role": "system", "content": system}]
            for name, text in self._history:
                role = "user" if name == self.config.user_name else "assistant"
                messages.append({"role": role, "content": text})
            return messages

    @classmethod
    def get_config_options(
        cls, field: str, values: dict[str, Any] | None = None
    ) -> list[dict[str, str]] | None:
        if field != "config.character_card":
            return None
        return [
            {"value": key, "label": card["title"]}
            for key, card in CHARACTER_CARDS.items()
        ]

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
                if has_memory and memory_sender is not None and memory_gen is not None:
                    memory_sender.send(
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
