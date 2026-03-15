from __future__ import annotations

import base64
import collections
import io
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionContentPartParam
from PIL import Image
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent, Tag
from src.core.frames import TextFrame, VideoFrame, VideoDataFormat


class StreamingVLMConfig(BaseModel):
    base_url: str
    api_key: str = ""
    model_id: str = "Qwen/Qwen3.5-0.8B"
    vlm_fps: int = 3
    window_duration: float = 3.0
    memory_size: int = 4
    key_window_interval: int | None = 4
    prompt_window1: str = (
        "You are an advanced real-time vision module for blind people. "
        "Given the current observation, use short phrases to caption what you see, include movement and composition if needed. "
        'Keep it short, efficient, real-time, relavant. Think "What would the blind person like to know?"'
    )
    prompt_window2_template: str = (
        "You are an advanced real-time vision module for blind people. "
        "Given the current observation history, generate an efficient delta caption "
        "only introducing new observations that were not mentioned in the history context. "
        'Keep it efficient, no redundant information, Think "What would the blind person like to know?"\nDO NOT REPEAT anything that is already mentioned.\n\n'
        "History context:\n{context}"
    )


class StreamingVLMInputs(NamedTuple):
    video: Receiver[VideoFrame]


class StreamingVLMOutputs(NamedTuple):
    observation: Sender[TextFrame]


class StreamingVLM(PrimitiveComponent[StreamingVLMInputs, StreamingVLMOutputs]):
    description = "Streams visual language model inference on video frames"

    """Streaming VLM component that generates captions from video frames.

    Single-threaded main loop consumes frames into a rolling buffer and
    submits VLM calls to a thread pool, polling for results each iteration.
    """

    _tags = Tag(io={"conduit"}, functionality={"video", "llm"}, gpu={"cpu", "nvidia", "apple"})

    def __init__(self, config: StreamingVLMConfig) -> None:
        super().__init__()
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url, api_key=config.api_key or "unused"
        )
        self._frame_buffer: collections.deque[tuple[float, VideoFrame]] = (
            collections.deque(maxlen=max(1, int(config.window_duration * 60)))
        )
        self._captions_history: list[str] = []

    def _encode_pil_to_base64_uri(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _should_use_key_window_prompt(self, window_index: int) -> bool:
        interval = self.config.key_window_interval
        return interval is not None and interval > 0 and window_index % interval == 0

    def _call_vllm(self, frames: list[Image.Image], prompt: str) -> str:
        content: list[ChatCompletionContentPartParam] = [
            {
                "type": "image_url",
                "image_url": {"url": self._encode_pil_to_base64_uri(img)},
            }
            for img in frames
        ]
        content.append({"type": "text", "text": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=512,
                temperature=0.7,
            )
            text = response.choices[0].message.content
            return text.strip() if text else ""
        except Exception as e:
            print(f"[StreamingVLM] VLLM call failed: {e}")
            return ""

    def _sample_window(self) -> list[Image.Image] | None:
        now = time.time()
        window_start = now - self.config.window_duration
        window_frames = [f for ts, f in self._frame_buffer if ts >= window_start]
        if not window_frames:
            return None

        frames_per_window = max(
            1, int(self.config.window_duration * self.config.vlm_fps)
        )
        sample_count = min(frames_per_window, len(window_frames))
        indices = np.linspace(0, len(window_frames) - 1, sample_count, dtype=int)

        return [
            Image.fromarray(window_frames[i].get(VideoDataFormat.RGB)) for i in indices
        ]

    def _build_prompt(self, window_count: int) -> str:
        if not self._captions_history or self._should_use_key_window_prompt(
            window_count
        ):
            return self.config.prompt_window1
        history_str = " ".join(self._captions_history)
        return self.config.prompt_window2_template.format(context=history_str)

    def _handle_caption(self, caption: str, outputs: StreamingVLMOutputs) -> None:
        if not caption:
            return
        print(f"[StreamingVLM] Caption: {caption}")
        self._captions_history.append(caption)
        if len(self._captions_history) > self.config.memory_size:
            self._captions_history.pop(0)
        outputs.observation.send(TextFrame.new(text=caption))

    def run(self, inputs: StreamingVLMInputs, outputs: StreamingVLMOutputs) -> None:
        print("[StreamingVLM] Starting Streaming VLM component")

        executor = ThreadPoolExecutor(max_workers=1)
        pending: Future[str] | None = None
        window_count = 0
        last_submit_time = 0.0

        for frame in inputs.video(self):
            if frame is None:
                break

            self._frame_buffer.append((time.time(), frame))

            # Check if pending VLM call is done
            if pending is not None and pending.done():
                self._handle_caption(pending.result(), outputs)
                pending = None

            # Submit new VLM call if none in flight and enough time has passed
            if (
                pending is None
                and time.time() - last_submit_time >= self.config.window_duration
            ):
                pil_frames = self._sample_window()
                if pil_frames:
                    window_count += 1
                    prompt = self._build_prompt(window_count)
                    pending = executor.submit(self._call_vllm, pil_frames, prompt)
                    last_submit_time = time.time()

        # Drain pending result
        if pending is not None:
            self._handle_caption(pending.result(), outputs)

        executor.shutdown(wait=False)
        print("[StreamingVLM] Streaming VLM component stopped")
