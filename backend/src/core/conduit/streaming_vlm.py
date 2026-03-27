from __future__ import annotations

import base64
import io
from typing import NamedTuple

from openai import OpenAI
from openai.types.chat import ChatCompletionContentPartParam
from PIL import Image
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import TextFrame, VideoFrame, VideoDataFormat


class VLMResponse(BaseModel):
    caption: str
    objects: str


class StreamingVLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    model_id: str = "gpt-5.4"
    memory_size: int = 4
    key_window_interval: int | None = 4
    prompt_window1: str = (
        "You are an advanced real-time vision module for blind people. "
        "Given the current observation, use short phrases to caption what you see, include movement and composition if needed. "
        'Keep it short, efficient, real-time, relavant. Think "What would the blind person like to know?" '
        "Also list the distinct objects/items visible as a unique comma-separated list in the objects field. No duplicates."
    )
    prompt_window2_template: str = (
        "You are an advanced real-time vision module for blind people. "
        "Given the current observation history, generate an efficient delta caption "
        "only introducing new observations that were not mentioned in the history context. "
        'Keep it efficient, no redundant information, Think "What would the blind person like to know?"\nDO NOT REPEAT anything that is already mentioned.\n\n'
        "Also list the distinct objects/items visible as a unique comma-separated list in the objects field. No duplicates.\n\n"
        "History context:\n{context}"
    )


class StreamingVLMInputs(NamedTuple):
    video: Receiver[VideoFrame]


class StreamingVLMOutputs(NamedTuple):
    observation: Sender[TextFrame]
    objects: Sender[TextFrame]


class StreamingVLM(ThreadedComponent[StreamingVLMInputs, StreamingVLMOutputs]):
    description = "Streams **visual language model** inference on `VideoFrame` input. Maintains a rolling frame buffer and submits VLM caption requests to a thread pool, outputting `TextFrame` descriptions."

    """Streaming VLM component that generates captions from video frames.

    Single-threaded main loop consumes frames into a rolling buffer and
    submits VLM calls to a thread pool, polling for results each iteration.
    """

    tags = Tag(
        io={"conduit"}, functionality={"video", "llm"}, gpu={"cpu", "nvidia", "apple"}
    )

    def __init__(self, config: StreamingVLMConfig) -> None:
        super().__init__()
        self.config = config
        self._client = OpenAI(base_url=config.base_url)
        self._captions_history: list[str] = []

    def _encode_pil_to_base64_uri(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _should_use_key_window_prompt(self, window_index: int) -> bool:
        interval = self.config.key_window_interval
        return interval is not None and interval > 0 and window_index % interval == 0

    def _call_vllm(
        self, frames: list[Image.Image], prompt: str
    ) -> tuple[str, str]:
        """Call VLM and return (caption, objects_csv)."""
        content: list[ChatCompletionContentPartParam] = [
            {
                "type": "image_url",
                "image_url": {"url": self._encode_pil_to_base64_uri(img)},
            }
            for img in frames
        ]
        content.append({"type": "text", "text": prompt})

        try:
            response = self._client.beta.chat.completions.parse(
                model=self.config.model_id,
                messages=[{"role": "user", "content": content}],
                response_format=VLMResponse,
                max_completion_tokens=512,
                temperature=0.7,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed.caption, parsed.objects
            # Fallback if parsing failed but content exists
            text = (response.choices[0].message.content or "").strip()
            return text, ""
        except Exception as e:
            print(f"[StreamingVLM] VLLM call failed: {e}")
            return "", ""

    def _build_prompt(self, window_count: int) -> str:
        if not self._captions_history or self._should_use_key_window_prompt(
            window_count
        ):
            return self.config.prompt_window1
        history_str = " ".join(self._captions_history)
        return self.config.prompt_window2_template.format(context=history_str)

    def _handle_result(
        self, caption: str, objects: str, outputs: StreamingVLMOutputs
    ) -> None:
        if caption:
            print(f"[StreamingVLM] Caption: {caption}")
            self._captions_history.append(caption)
            if len(self._captions_history) > self.config.memory_size:
                self._captions_history.pop(0)
            outputs.observation.send(TextFrame.new(text=caption))
        if objects:
            print(f"[StreamingVLM] Objects: {objects}")
            outputs.objects.send(TextFrame.new(text=objects))

    def run(self, inputs: StreamingVLMInputs, outputs: StreamingVLMOutputs) -> None:
        print("[StreamingVLM] Starting Streaming VLM component")

        inputs.video.newest = True

        window_count = 0

        for frame in inputs.video:
            if frame is None:
                break

            pil_img = Image.fromarray(frame.get(VideoDataFormat.RGB))
            window_count += 1
            prompt = self._build_prompt(window_count)
            caption, objects = self._call_vllm([pil_img], prompt)
            self._handle_result(caption, objects, outputs)

        print("[StreamingVLM] Streaming VLM component stopped")
