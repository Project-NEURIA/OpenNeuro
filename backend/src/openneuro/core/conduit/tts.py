from __future__ import annotations

import base64
import json
import os
import re
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue

import requests

from openneuro.config import get_config_value
from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import AudioFrame, InterruptFrame, TextFrame

# Text processing utilities
CutFn = "Callable[[str], int]"


def cut_space(buf: str) -> int:
    return max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))


_SENT_END = re.compile(r"""[.!?…]+["'\)\]\}]*($|\s)""")


def cut_sentence(buf: str) -> int:
    buf.strip()
    last = -1
    for m in _SENT_END.finditer(buf):
        cut = m.end() - 1
        while cut >= 0 and buf[cut].isspace():
            cut -= 1
        last = max(last, cut)
    return last


def cut_regex(pattern: str) -> CutFn:
    rx = re.compile(pattern)
    def _cut(buf: str) -> int:
        last = -1
        for m in rx.finditer(buf):
            last = max(last, m.end() - 1)
        return last
    return _cut


def cut_sentence_or_len(n: int = 80) -> CutFn:
    def _cut(buf: str) -> int:
        s = cut_sentence(buf)
        return s if s >= 0 else (n - 1 if len(buf) >= n else -1)
    return _cut


@dataclass
class StreamFilter:
    """Filters text to remove markdown and formatting for TTS."""
    speak_buf: str = ""
    in_square: int = 0
    in_paren: int = 0
    in_angle: int = 0
    in_bold: bool = False
    in_italic: bool = False
    cut_fn: CutFn = field(default=cut_space)

    def __init__(self):
        self.set_mode("sentence")

    def set_mode(self, mode: str) -> None:
        if mode == "space":
            self.cut_fn = cut_space
        elif mode == "sentence":
            self.cut_fn = cut_sentence
        elif mode.startswith("regex:"):
            self.cut_fn = cut_regex(mode[len("regex:"):])
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def feed(self, token: str, force: bool = False) -> str:
        self._consume(token)
        if force:
            out, self.speak_buf = self.speak_buf, ""
            return out

        cut = self.cut_fn(self.speak_buf)
        if cut < 0:
            return ""
        out = self.speak_buf[: cut + 1]
        self.speak_buf = self.speak_buf[cut + 1 :]
        return out

    def _consume(self, token: str) -> None:
        i = 0
        while i < len(token):
            ch = token[i]

            if ch == "*" and i + 1 < len(token) and token[i + 1] == "*":
                self.in_bold = not self.in_bold
                i += 2
                continue
            if ch == "*":
                self.in_italic = not self.in_italic
                i += 1
                continue

            if not (self.in_bold or self.in_italic):
                if ch == "[":
                    self.in_square += 1; i += 1; continue
                if ch == "]" and self.in_square:
                    self.in_square -= 1; i += 1; continue
                if ch == "(":
                    self.in_paren += 1; i += 1; continue
                if ch == ")" and self.in_paren:
                    self.in_paren -= 1; i += 1; continue
                if ch == "<":
                    self.in_angle += 1; i += 1; continue
                if ch == ">" and self.in_angle:
                    self.in_angle -= 1; i += 1; continue

            if (self.in_square or self.in_paren or self.in_angle or self.in_bold or self.in_italic):
                i += 1
                continue

            self.speak_buf += ch
            i += 1


GENERATE_END_FLAG = "[END_OF_GENERATE]"


class TTS(Component[Channel[TextFrame]]):
    """Text-to-Speech component using Inworld API.
    
    Takes text chunks, filters them, and streams audio back.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        cut_mode: str = "sentence",
    ) -> None:
        super().__init__()
        self._output = Channel[AudioFrame]()
        
        # Load from config if not provided
        tts_config = get_config_value("tts", {})
        self.url = url or tts_config.get("url", "https://api.inworld.ai/v1/tts/stream")
        self.voice_id = voice_id or tts_config.get("voice_id", "default")
        self.model_id = model_id or tts_config.get("model_id", "default")
        self.cut_mode = cut_mode
        
        # Stream filter for text processing
        self._stream_filter = StreamFilter()
        self._stream_filter.set_mode(cut_mode)
        
        # Generation tracking
        self._generation = 0
        self._gen_lock = threading.Lock()
        
        # Task queue for worker thread
        self._task_queue: Queue[tuple[int, str]] = Queue()
        self._worker_thread: threading.Thread | None = None

    def get_output_channels(self) -> tuple[Channel[AudioFrame]]:
        return (self._output,)

    def set_input_channels(self, input_channel: Channel[TextFrame]) -> None:
        self._input_channel = input_channel

    def _worker(self) -> None:
        """Worker thread that processes TTS requests."""
        print("[TTS] Worker thread started")
        
        while not self.stop_event.is_set():
            try:
                gen, text = self._task_queue.get(timeout=0.1)
                
                # Check if this generation is still current
                with self._gen_lock:
                    if gen != self._generation:
                        print(f"[TTS] Generation {gen} cancelled, current is {self._generation}")
                        continue
                
                # Get API credentials
                cred = os.getenv("INWORLD_API_CRED")
                if not cred:
                    print("[TTS] INWORLD_API_CRED not set")
                    continue
                
                headers = {
                    "Authorization": f"Basic {cred}",
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "text": text,
                    "voiceId": self.voice_id,
                    "modelId": self.model_id,
                    "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000},
                }
                
                print(f"[TTS] Sending TTS request for: {text[:50]}...")
                
                try:
                    r = requests.post(self.url, json=payload, headers=headers, stream=True, timeout=10)
                    r.raise_for_status()
                    
                    canceled_during_generation = False
                    for line in r.iter_lines():
                        print(f"[TTS] Received line {line}")
                        # Check if generation was cancelled
                        with self._gen_lock:
                            if gen != self._generation:
                                print("[TTS] Job interrupted, stopping TTS stream")
                                canceled_during_generation = True
                                break
                        
                        if not line:
                            continue
                        
                        msg = json.loads(line)
                        raw = base64.b64decode(msg["result"]["audioContent"])
                        if len(raw) > 44:
                            pcm = raw[44:]  # Skip WAV header
                            self._output.send(AudioFrame(
                                frame_type_string="tts_audio",
                                pcm16_data=pcm,
                                sample_rate=48000,
                                channels=1
                            ))
                    
                    if not canceled_during_generation:
                        # Send completion text frame
                        self._output.send(TextFrame(
                            frame_type_string="tts_text",
                            text=text,
                            language="en"
                        ))
                        
                except requests.exceptions.RequestException as e:
                    print(f"[TTS] API request failed: {e}")
                except Exception as e:
                    print(f"[TTS] Generation error: {e}")
                    
            except Empty:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[TTS] Worker error: {e}")
                continue

    def run(self) -> None:
        """Main processing loop."""
        print("[TTS] Starting TTS")
        
        # Wait for input channel
        while self._input_channel is None and not self.stop_event.is_set():
            import time
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
                print(f"[TTS] Interrupt received: {frame.reason}")
                with self._gen_lock:
                    self._generation += 1
                # Reset stream filter
                self._stream_filter = StreamFilter()
                self._stream_filter.set_mode(self.cut_mode)
                # Clear task queue
                while not self._task_queue.empty():
                    try:
                        self._task_queue.get_nowait()
                    except Empty:
                        break
                continue
            
            # Only process text chunks
            if frame.frame_type_string != "llm_chunk":
                continue
            
            t = frame.text
            out = self._stream_filter.feed("", force=True) if t == GENERATE_END_FLAG else self._stream_filter.feed(t)
            
            # Submit text for TTS if not empty
            if out and out.strip():
                with self._gen_lock:
                    gen = self._generation
                self._task_queue.put((gen, out))
                print(f"[TTS] Queued text: {out[:50]}...")
        
        # Clean up worker thread
        if self._worker_thread:
            self._worker_thread.join(timeout=1)
        
        print("[TTS] TTS stopped")
