from __future__ import annotations

import collections
import threading
import time
from typing import NamedTuple

import base64
import io

import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import TextFrame, VideoFrame, VideoDataFormat


class StreamingVLMConfig(BaseModel):
    vllm_url: str = "http://localhost:8055/v1/chat/completions"
    model_id: str = "Qwen/Qwen3.5-0.8B"
    vlm_fps: int = 3
    window_duration: float = 3.0
    memory_size: int = 4
    key_window_interval: int | None = 4
    prompt_window1: str = (
        "You are an advanced real-time vision module for blind people. "
        "Given the current observation, use short phrases to caption what you see, include movement and composition if needed. "
        "Keep it short, efficient, real-time, relavant. Think \"What would the blind person like to know?\""
    )
    prompt_window2_template: str = (
        "You are an advanced real-time vision module for blind people. "
        "Given the current observation history, generate an efficient delta caption "
        "only introducing new observations that were not mentioned in the history context. "
        "Keep it efficient, no redundant information, Think \"What would the blind person like to know?\"\nDO NOT REPEAT anything that is already mentioned.\n\n"
        "History context:\n{context}"
    )


class StreamingVLMInputs(NamedTuple):
    video: Receiver[VideoFrame]


class StreamingVLMOutputs(NamedTuple):
    observation: Sender[TextFrame]


class StreamingVLM(Component[StreamingVLMInputs, StreamingVLMOutputs]):
    """Streaming VLM component that generates captions from video frames.
    
    It maintains a rolling buffer of frames and periodically sends them to a VLM
    to generate delta captions based on previous observations.
    """

    def __init__(self, config: StreamingVLMConfig) -> None:
        super().__init__()
        self.config = config
        self._frame_buffer: collections.deque[tuple[float, VideoFrame]] = (
            collections.deque(maxlen=max(1, int(config.window_duration * 60)))
        )
        self._buffer_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._captions_history: list[str] = []
        self._history_lock = threading.Lock()

    def _get_latest_window_frames(self) -> tuple[list[VideoFrame], float | None]:
        now = time.time()
        window_start = now - self.config.window_duration
        with self._buffer_lock:
            snapshot = list(self._frame_buffer)

        if not snapshot:
            return [], None

        window_frames = [frame for ts, frame in snapshot if ts >= window_start]
        if not window_frames:
            return [], snapshot[-1][0]

        latest_timestamp = snapshot[-1][0]
        return window_frames, latest_timestamp

    def _encode_pil_to_base64_uri(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _should_use_key_window_prompt(self, window_index: int) -> bool:
        interval = self.config.key_window_interval
        return interval is not None and interval > 0 and window_index % interval == 0

    def _call_vllm(self, frames: list[Image.Image], prompt: str) -> str:
        content = []
        for img in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_pil_to_base64_uri(img)}
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "temperature": 0.7,
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(self.config.vllm_url, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                print(f"[StreamingVLM] Error {response.status_code}: {response.text}")
                return ""
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[StreamingVLM] VLLM call failed: {e}")
            return ""

    def run(self, inputs: StreamingVLMInputs, outputs: StreamingVLMOutputs) -> None:
        print("[StreamingVLM] Starting Streaming VLM component")

        def capture_frames():
            for frame in inputs.video(self):
                if frame is None:
                    break
                with self._buffer_lock:
                    self._frame_buffer.append((time.time(), frame))
                self._new_frame_event.set()

        def process_vlm():
            frames_per_window = max(1, int(self.config.window_duration * self.config.vlm_fps))
            last_processed_latest_ts: float | None = None
            processed_window_count = 0
            
            while not self.stop_event.is_set():
                self._new_frame_event.wait(timeout=0.1)
                self._new_frame_event.clear()

                window_frames_data, latest_timestamp = self._get_latest_window_frames()
                if not window_frames_data:
                    continue
                if latest_timestamp is None or latest_timestamp == last_processed_latest_ts:
                    continue

                sample_count = min(frames_per_window, len(window_frames_data))
                indices = np.linspace(
                    0,
                    len(window_frames_data) - 1,
                    sample_count,
                    dtype=int,
                )
                sampled_frames = [window_frames_data[i] for i in indices]
                
                pil_frames = []
                for f in sampled_frames:
                    rgb = f.get(VideoDataFormat.RGB)
                    pil_frames.append(Image.fromarray(rgb))

                processed_window_count += 1
                with self._history_lock:
                    if (
                        not self._captions_history
                        or self._should_use_key_window_prompt(processed_window_count)
                    ):
                        prompt = self.config.prompt_window1
                    else:
                        history_str = " ".join(self._captions_history)
                        prompt = self.config.prompt_window2_template.format(context=history_str)
                
                caption = self._call_vllm(pil_frames, prompt)
                last_processed_latest_ts = latest_timestamp
                if caption:
                    print(f"[StreamingVLM] Caption: {caption}")
                    with self._history_lock:
                        self._captions_history.append(caption)
                        if len(self._captions_history) > self.config.memory_size:
                            self._captions_history.pop(0)
                        
                        # Send the latest observation
                        outputs.observation.send(TextFrame.new(text=caption))

        threads = [
            threading.Thread(target=capture_frames, daemon=True),
            threading.Thread(target=process_vlm, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        print("[StreamingVLM] Streaming VLM component stopped")
