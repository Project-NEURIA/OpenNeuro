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


class CaptionResponse(BaseModel):
    caption: str


class ObjectsResponse(BaseModel):
    objects: str


class StreamingVLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    model_id: str = "gpt-5.4"
    max_history: int = 3
    caption_prompt: str = (
        "You are a real-time vision module. "
        "If there are previous frames, describe only what is NEW or CHANGED. "
        "Do not repeat what was already described. "
        "If this is the first frame, describe the full scene."
    )
    objects_prompt: str = (
        "You are a real-time vision module tracking visible objects. "
        "Given the latest caption and the current active objects list, "
        "output the updated active objects list reflecting what is currently visible. "
        "Maintain consistency with the existing list — keep the same naming for objects that are still present, "
        "add new ones, and remove ones no longer visible. "
        "Output as a single comma-separated string in the objects field."
    )


class StreamingVLMInputs(NamedTuple):
    video: Receiver[VideoFrame]


class StreamingVLMOutputs(NamedTuple):
    observation: Sender[TextFrame]
    objects: Sender[TextFrame]


class StreamingVLM(ThreadedComponent[StreamingVLMInputs, StreamingVLMOutputs]):
    description = (
        "Real-time vision module that captions video frames and tracks visible objects."
    )

    tags = Tag(
        io={"conduit"}, functionality={"video", "llm"}, gpu={"cpu", "nvidia", "apple"}
    )

    def __init__(self, config: StreamingVLMConfig) -> None:
        super().__init__()
        self.config = config
        self._client = OpenAI(base_url=config.base_url)
        self._history: list[dict] = []  # [{image_uri, caption}]
        self._active_objects: str = ""

    def _encode_pil_to_base64_uri(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _call_caption(self, current_image_uri: str) -> str | None:
        msgs: list[dict] = [{"role": "system", "content": self.config.caption_prompt}]

        # History: oldest to newest, each is image + caption pair
        for entry in self._history:
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": entry["image_uri"]},
                        },
                    ],
                }
            )
            msgs.append(
                {
                    "role": "assistant",
                    "content": entry["caption"],
                }
            )

        # Current frame
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": current_image_uri},
                    },
                ],
            }
        )

        try:
            response = self._client.beta.chat.completions.parse(
                model=self.config.model_id,
                messages=msgs,  # type: ignore[arg-type]
                response_format=CaptionResponse,
                max_completion_tokens=512,
            )
            parsed = response.choices[0].message.parsed
            return parsed.caption if parsed else None
        except Exception as e:
            print(f"[StreamingVLM] Caption call failed: {e}")
            return None

    def _call_objects(self, caption: str) -> str | None:
        msgs: list[dict] = [
            {"role": "system", "content": self.config.objects_prompt},
            {
                "role": "user",
                "content": (
                    f"Latest caption: {caption}\n\n"
                    f"Current active objects: {self._active_objects or '(none)'}"
                ),
            },
        ]

        try:
            response = self._client.beta.chat.completions.parse(
                model=self.config.model_id,
                messages=msgs,  # type: ignore[arg-type]
                response_format=ObjectsResponse,
                max_completion_tokens=256,
            )
            parsed = response.choices[0].message.parsed
            return parsed.objects if parsed else None
        except Exception as e:
            print(f"[StreamingVLM] Objects call failed: {e}")
            return None

    def run(self, inputs: StreamingVLMInputs, outputs: StreamingVLMOutputs) -> None:
        print("[StreamingVLM] Starting Streaming VLM component")

        inputs.video.newest = True

        for frame in inputs.video:
            if frame is None:
                break

            pil_img = Image.fromarray(frame.get(VideoDataFormat.RGB))
            image_uri = self._encode_pil_to_base64_uri(pil_img)

            # Call 1: caption
            caption = self._call_caption(image_uri)
            if caption is None:
                continue

            # Update history
            self._history.append({"image_uri": image_uri, "caption": caption})
            if len(self._history) > self.config.max_history:
                self._history.pop(0)

            print(f"[StreamingVLM] Caption: {caption}")
            outputs.observation.send(TextFrame.new(text=caption))

            # Call 2: objects
            new_objects = self._call_objects(caption)
            if new_objects is not None:
                self._active_objects = new_objects
                print(f"[StreamingVLM] Objects: {self._active_objects}")
                outputs.objects.send(TextFrame.new(text=self._active_objects))

        print("[StreamingVLM] Streaming VLM component stopped")
