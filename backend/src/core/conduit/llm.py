from __future__ import annotations

import json
import os
import threading
from queue import Empty, Queue
from typing import NamedTuple

import requests
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import InterruptFrame, MessagesFrame, TextFrame

GENERATE_END_FLAG = "[END_OF_GENERATE]"


class LLMConfig(BaseModel):
    url: str = "https://api.groq.com/openai/v1/chat/completions"
    model_id: str = "llama-3.3-70b-versatile"
    api_key_env_var: str = "GROQ_API_KEY"
    top_p: float = 0.97
    temperature: float = 1.08
    max_tokens: int = 350


class LLMInputs(NamedTuple):
    messages: Receiver[MessagesFrame]
    interrupt: Receiver[InterruptFrame] | None = None


class LLMOutputs(NamedTuple):
    text: Sender[TextFrame]
    interrupt: Sender[InterruptFrame]


class LLM(Component[LLMInputs, LLMOutputs]):
    """LLM text generation component using Groq API."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config: LLMConfig = config

        # Generation tracking
        self._generation = 0
        self._gen_lock = threading.Lock()

        # Task queue for worker thread
        self._task_queue: Queue[tuple[int, MessagesFrame]] = Queue()

    def _worker(self, outputs: LLMOutputs) -> None:
        print("[LLM] Worker thread started")
        while not self.stop_event.is_set():
            try:
                gen, frame = self._task_queue.get(timeout=0.1)

                with self._gen_lock:
                    if gen != self._generation:
                        continue

                self._process_generation(gen, frame, outputs)
            except Empty:
                continue
            except Exception as e:
                print(f"[LLM] Worker error: {e}")

    def run(self, inputs: LLMInputs, outputs: LLMOutputs) -> None:
        print("[LLM] Starting LLM generation")

        worker_thread = threading.Thread(
            target=self._worker, args=(outputs,), daemon=True
        )
        worker_thread.start()

        if inputs.interrupt is not None:
            interrupt_recv = inputs.interrupt

            def handle_interrupts() -> None:
                for frame in interrupt_recv(self):
                    if frame is None:
                        break
                    print(f"[LLM] Interrupt received: {frame.reason}")

                    # Signal interruption to generation loop
                    with self._gen_lock:
                        self._generation += 1

                    # Forward the interrupt
                    outputs.interrupt.send(frame)

                    # Clear queue
                    while not self._task_queue.empty():
                        try:
                            self._task_queue.get_nowait()
                        except Empty:
                            break

            threading.Thread(target=handle_interrupts, daemon=True).start()

        for frame in inputs.messages(self):
            if frame is None:
                break

            with self._gen_lock:
                gen = self._generation
            self._task_queue.put((gen, frame))

        worker_thread.join(timeout=1)
        print("[LLM] LLM generation stopped")

    def _process_generation(
        self, gen: int, frame: MessagesFrame, outputs: LLMOutputs
    ) -> None:
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            raise ValueError(f"Environment variable {self.config.api_key_env_var} must be set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_id,
            "messages": frame.messages,
            "stream": True,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "stop": ["\n"],
            "max_tokens": self.config.max_tokens,
        }

        try:
            r = requests.post(
                self.config.url, headers=headers, json=payload, stream=True, timeout=60
            )
            r.raise_for_status()

            for line in r.iter_lines():
                with self._gen_lock:
                    if gen != self._generation:
                        outputs.text.send(TextFrame.new(text=GENERATE_END_FLAG))
                        break

                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                choice = choices[0]
                if choice.get("finish_reason"):
                    outputs.text.send(TextFrame.new(text=GENERATE_END_FLAG))
                    break

                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    outputs.text.send(TextFrame.new(text=text))

        except Exception as e:
            print(f"[LLM] Generation error: {e}")
