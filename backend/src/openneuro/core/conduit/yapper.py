from __future__ import annotations

import json
import os
import threading
import time
from queue import Empty, Queue

import requests

from openneuro.config import get_config_value
from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import ConversationalFrame, InterruptFrame, TextFrame

GENERATE_END_FLAG = "[END_OF_GENERATE]"


class Yapper(Component[Channel[ConversationalFrame]]):
    """LLM text generation component using Groq API.
    
    Takes ConversationalFrame with context/messages and streams text chunks.
    Handles interrupts and generation cancellation.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        model_id: str | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._output = Channel[TextFrame]()
        
        # Load from config if not provided
        yapper_config = get_config_value("yapper_llm", {})
        self.url = url or yapper_config.get("url", "https://api.groq.com/openai/v1/chat/completions")
        self.model_id = model_id or yapper_config.get("model_id", "llama3-8b-8192")
        self.top_p = top_p if top_p is not None else yapper_config.get("top_p", 0.97)
        self.temperature = temperature if temperature is not None else yapper_config.get("temperature", 1.08)
        self.max_tokens = max_tokens if max_tokens is not None else yapper_config.get("max_tokens", 350)
        
        # Generation tracking
        self._generation = 0
        self._gen_lock = threading.Lock()
        
        # Task queue for worker thread
        self._task_queue: Queue[tuple[int, ConversationalFrame]] = Queue()
        self._worker_thread: threading.Thread | None = None

    def get_output_channels(self) -> tuple[Channel[TextFrame]]:
        return (self._output,)

    def set_input_channels(self, input_channel: Channel[ConversationalFrame]) -> None:
        self._input_channel = input_channel

    def _is_chat_completions(self) -> bool:
        """True if URL is /v1/chat/completions (uses messages)."""
        return "/v1/chat/completions" in self.url

    def _worker(self) -> None:
        """Worker thread that processes LLM generation tasks."""
        print("[Yapper] Worker thread started")
        
        while not self.stop_event.is_set():
            try:
                gen, frame = self._task_queue.get(timeout=0.1)
                
                # Check if this generation is still current
                with self._gen_lock:
                    if gen != self._generation:
                        print(f"[Yapper] Generation {gen} cancelled, current is {self._generation}")
                        continue
                
                # Process the generation
                self._process_generation(gen, frame)
                
            except Empty:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Yapper] Worker error: {e}")
                continue

    def run(self) -> None:
        """Main processing loop."""
        print("[Yapper] Starting LLM generation")
        
        # Wait for input channel
        while self._input_channel is None and not self.stop_event.is_set():
            time.sleep(0.1)
        
        if self.stop_event.is_set():
            return
        
        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        
        # Process input frames
        for frame in self._input_channel.stream(self.stop_event):
            if frame is None:
                break
            
            # Handle interrupts
            if isinstance(frame, InterruptFrame):
                print(f"[Yapper] Interrupt received: {frame.reason}")
                with self._gen_lock:
                    self._generation += 1
                # Clear task queue
                while not self._task_queue.empty():
                    try:
                        self._task_queue.get_nowait()
                    except Empty:
                        break
                continue
            
            print(f"[Yapper] Received frame: {frame.frame_type_string}")
            
            # Only process agent_state frames (from AgentState component)
            if frame.frame_type_string != "agent_state":
                continue
            
            # Get current generation ID
            with self._gen_lock:
                gen = self._generation
            
            # Queue task for worker thread (generation happens there)
            self._task_queue.put((gen, frame))
        
        # Clean up worker thread
        if self._worker_thread:
            self._worker_thread.join(timeout=1)
        
        print("[Yapper] LLM generation stopped")

    def _process_generation(self, gen: int, frame: ConversationalFrame) -> None:
        """Process a single LLM generation."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[Yapper] GROQ_API_KEY not set")
            return
        
        use_chat = self._is_chat_completions()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        if use_chat:
            payload = {
                "model": self.model_id,
                "messages": frame.messages,
                "stream": True,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "stop": ["\n"],
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": self.max_tokens,
            }
        else:
            payload = {
                "model": self.model_id,
                "prompt": frame.text,
                "stream": True,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "stop": ["\n"],
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": self.max_tokens,
            }
        
        print(f"[Yapper] Sending request to {self.url}")
        
        try:
            r = requests.post(self.url, headers=headers, json=payload, stream=True, timeout=60)
            r.raise_for_status()
            
            for line in r.iter_lines():
                with self._gen_lock:
                    if gen != self._generation:
                        self._output.send(TextFrame(
                            frame_type_string="llm_chunk",
                            text=GENERATE_END_FLAG,
                            language="en"
                        ))
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
                finish_reason = choice.get("finish_reason")
                print(f"[Yapper] {choice}")
                if finish_reason is not None:
                    self._output.send(TextFrame(
                        frame_type_string="llm_chunk",
                        text=GENERATE_END_FLAG,
                        language="en"
                    ))
                    break
                
                if use_chat:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                else:
                    text = choice.get("text") or ""
                
                if text:
                    self._output.send(TextFrame(
                        frame_type_string="llm_chunk",
                        text=text,
                        language="en"
                    ))
                    
        except requests.exceptions.RequestException as e:
            print(f"[Yapper] API request failed: {e}")
        except Exception as e:
            print(f"[Yapper] Generation error: {e}")
