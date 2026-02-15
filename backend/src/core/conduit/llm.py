from __future__ import annotations

import json
import os
import threading
from queue import Empty, Queue
from typing import TypedDict

import requests

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import MessagesFrame, InterruptFrame, TextFrame, MessagesDataFormat
from src.core.config import BaseConfig

GENERATE_END_FLAG = "[END_OF_GENERATE]"


class LLMConfig(BaseConfig):
    url: str = "https://api.groq.com/openai/v1/chat/completions"
    model_id: str = "llama3-8b-8192"
    top_p: float = 0.97
    temperature: float = 1.08
    max_tokens: int = 350


class LLMOutputs(TypedDict):
    text: Channel[TextFrame]
    interrupt: Channel[InterruptFrame]


class LLM(Component[[Channel[MessagesFrame], Channel[InterruptFrame]], LLMOutputs]):
    """LLM text generation component using Groq API."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config or LLMConfig())
        self.config: LLMConfig
        self._output_text = Channel[TextFrame](name="text")
        self._output_interrupt = Channel[InterruptFrame](name="interrupt")
        
        # Generation tracking
        self._generation = 0
        self._gen_lock = threading.Lock()
        
        # Task queue for worker thread
        self._task_queue: Queue[tuple[int, MessagesFrame]] = Queue()

    def get_output_channels(self) -> LLMOutputs:
        return {
            "text": self._output_text,
            "interrupt": self._output_interrupt
        }

    def _worker(self) -> None:
        print("[LLM] Worker thread started")
        while not self.stop_event.is_set():
            try:
                gen, frame = self._task_queue.get(timeout=0.1)
                
                with self._gen_lock:
                    if gen != self._generation:
                        continue
                
                self._process_generation(gen, frame)
            except Empty:
                continue
            except Exception as e:
                print(f"[LLM] Worker error: {e}")

    def run(self, messages: Channel[MessagesFrame] | None = None, interrupt: Channel[InterruptFrame] | None = None) -> None:
        print("[LLM] Starting LLM generation")
        
        worker_thread = threading.Thread(target=self._worker, daemon=True)
        worker_thread.start()

        def handle_interrupts():
            if not interrupt: return
            for frame in interrupt.stream(self):
                if frame is None: break
                print(f"[LLM] Interrupt received: {frame.get()}")
                
                # Signal interruption to generation loop
                with self._gen_lock:
                    self._generation += 1
                
                # Forward the interrupt
                self._output_interrupt.send(frame)
                
                # Clear queue
                while not self._task_queue.empty():
                    try: self._task_queue.get_nowait()
                    except Empty: break
                    
        threading.Thread(target=handle_interrupts, daemon=True).start()

        if messages:
            for frame in messages.stream(self):
                if frame is None: break
                
                with self._gen_lock:
                    gen = self._generation
                self._task_queue.put((gen, frame))
        
        worker_thread.join(timeout=1)
        print("[LLM] LLM generation stopped")

    def _process_generation(self, gen: int, frame: MessagesFrame) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[LLM] GROQ_API_KEY not set")
            return
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.model_id,
            "messages": frame.get(MessagesDataFormat.MESSAGES),
            "stream": True,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "stop": ["\n"],
            "max_tokens": self.config.max_tokens,
        }
        
        try:
            r = requests.post(self.config.url, headers=headers, json=payload, stream=True, timeout=60)
            r.raise_for_status()
            
            for line in r.iter_lines():
                with self._gen_lock:
                    if gen != self._generation:
                        self._output_text.send(TextFrame(display_name="llm_chunk", text=GENERATE_END_FLAG))
                        break
                
                if not line: continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "): continue
                
                data_str = decoded[6:].strip()
                if data_str == "[DONE]": break
                
                try: chunk = json.loads(data_str)
                except json.JSONDecodeError: continue
                
                choices = chunk.get("choices") or []
                if not choices: continue
                
                choice = choices[0]
                if choice.get("finish_reason"):
                    self._output_text.send(TextFrame(display_name="llm_chunk", text=GENERATE_END_FLAG))
                    break
                
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    self._output_text.send(TextFrame(display_name="llm_chunk", text=text))
                    
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
