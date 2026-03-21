from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal, overload, NamedTuple

import numpy as np

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from src.core.utils import obj_id


class BonePose(NamedTuple):
    """Position (meters) + quaternion rotation (w,x,y,z) for a single bone."""

    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    rot_w: float = 1.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0


class AudioDataFormat(Enum):
    PCM8 = "pcm8"
    PCM16 = "pcm16"
    FLOAT32 = "float32"


@dataclass(frozen=True, slots=True)
class Frame:
    """Base frame class for all frames in the pipeline."""

    pts: int
    id: int

    def __str__(self) -> str:
        return f"{type(self).__name__}(id={self.id}, pts={self.pts})"


@dataclass(frozen=True, slots=True)
class AudioFrame(Frame):
    """Audio frame with immutable data and on-the-fly reformatting/resampling."""

    data: np.ndarray
    sample_rate: int
    channels: int

    @classmethod
    def new(
        cls,
        *,
        data: bytes | np.ndarray,
        sample_rate: int,
        channels: int = 1,
    ) -> AudioFrame:
        # Normalize data to np.ndarray shape (channels, samples) float32
        # PCM bytes and 1D arrays are assumed INTERLEAVED: [L0,R0,L1,R1,...]
        if isinstance(data, bytes):
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                arr = arr.reshape(-1, channels).T  # deinterleave
            else:
                arr = arr.reshape(1, -1)
        elif isinstance(data, np.ndarray):
            if data.dtype == np.int16:
                arr = data.astype(np.float32) / 32768.0
            else:
                arr = data.astype(np.float32)

            if arr.ndim == 1:
                if channels > 1:
                    arr = arr.reshape(-1, channels).T  # deinterleave
                else:
                    arr = arr.reshape(1, -1)
            elif arr.ndim == 2:
                # Already 2D: if shape is (samples, channels), transpose to (channels, samples)
                if arr.shape[0] != channels and arr.shape[1] == channels:
                    arr = arr.T
                # else assume already (channels, samples)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            data=arr,
            sample_rate=sample_rate,
            channels=channels,
        )

    @overload
    def get(
        self,
        data_format: Literal[AudioDataFormat.FLOAT32],
        sample_rate: int | None = None,
        num_channels: int | None = None,
    ) -> np.ndarray: ...

    @overload
    def get(
        self,
        data_format: Literal[AudioDataFormat.PCM16, AudioDataFormat.PCM8],
        sample_rate: int | None = None,
        num_channels: int | None = None,
    ) -> bytes: ...

    def get(
        self,
        data_format: AudioDataFormat,
        sample_rate: int | None = None,
        num_channels: int | None = None,
    ) -> np.ndarray | bytes:
        """Get the audio data in the requested format, sample rate, and channels."""
        arr = self.data
        current_sr = self.sample_rate
        current_ch = self.channels

        # 1. Resample if needed
        if sample_rate and sample_rate != current_sr:
            num_samples = int(arr.shape[1] * sample_rate / current_sr)
            arr = np.stack(
                [
                    np.interp(
                        np.linspace(0, arr.shape[1], num_samples, endpoint=False),
                        np.arange(arr.shape[1]),
                        ch_data,
                    )
                    for ch_data in arr
                ]
            )

        # 2. Change channels if needed
        if num_channels and num_channels != current_ch:
            if num_channels == 1:
                arr = arr.mean(axis=0, keepdims=True)
            elif num_channels == 2 and current_ch == 1:
                arr = np.vstack([arr, arr])
            else:
                if num_channels < current_ch:
                    arr = arr[:num_channels, :]
                else:
                    padding = np.zeros((num_channels - current_ch, arr.shape[1]))
                    arr = np.vstack([arr, padding])

        # 3. Format conversion
        # arr is (channels, samples) — transpose to (samples, channels) for interleaved output
        if data_format == AudioDataFormat.FLOAT32:
            return arr

        if data_format == AudioDataFormat.PCM16:
            interleaved = arr.T.flatten() if arr.shape[0] > 1 else arr.flatten()
            return (
                np.clip(interleaved * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            )

        if data_format == AudioDataFormat.PCM8:
            interleaved = arr.T.flatten() if arr.shape[0] > 1 else arr.flatten()
            return (
                np.clip((interleaved + 1.0) * 127.5, 0, 255).astype(np.uint8).tobytes()
            )

        raise ValueError(f"Unsupported data format: {data_format}")


@dataclass(frozen=True, slots=True)
class TextFrame(Frame):
    """Frame containing text data."""

    text: str
    language: str | None = None

    @classmethod
    def new(cls, *, text: str, language: str | None = None) -> TextFrame:
        return cls(pts=time.time_ns(), id=obj_id(), text=text, language=language)

    def get(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class EOS(TextFrame):
    """End-of-sequence sentinel. Subclasses TextFrame so Receiver[TextFrame] accepts it."""

    END: ClassVar[EOS]  # type: ignore[misc]

    @classmethod
    def new(cls, *, text: str = "", language: str | None = None) -> EOS:
        return cls(pts=0, id=0, text=text, language=language)


EOS.END = EOS.new()


@dataclass(frozen=True, slots=True)
class InterruptFrame(Frame):
    """Frame representing an interrupt event."""

    reason: str

    @classmethod
    def new(
        cls,
        *,
        reason: str,
    ) -> InterruptFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class RequestFrame(Frame):
    """Frame that triggers a response from the agent."""

    @classmethod
    def new(cls) -> RequestFrame:
        return cls(pts=time.time_ns(), id=obj_id())


@dataclass(frozen=True, slots=True)
class MessageFrame(Frame):
    """A single chat message (OpenAI-compatible)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    tool_call_id: str | None = None

    @classmethod
    def new(
        cls,
        *,
        role: Literal["system", "user", "assistant", "tool"],
        content: str | None = None,
        tool_calls: list[ChatCompletionMessageToolCall] | None = None,
        tool_call_id: str | None = None,
    ) -> MessageFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )


@dataclass(frozen=True, slots=True)
class ToolDef(Frame):
    """Definition of a tool that an LLM can call (matches OpenAI FunctionDefinition)."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None

    @classmethod
    def new(
        cls,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        strict: bool | None = None,
    ) -> ToolDef:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            name=name,
            description=description,
            parameters=parameters,
            strict=strict,
        )


@dataclass(frozen=True, slots=True)
class ToolCall(Frame):
    """A tool call emitted by an LLM (matches OpenAI ChatCompletionMessageToolCall)."""

    call_id: str
    name: str
    arguments: str

    @classmethod
    def new(cls, *, call_id: str, name: str, arguments: str) -> ToolCall:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            call_id=call_id,
            name=name,
            arguments=arguments,
        )


@dataclass(frozen=True, slots=True)
class ToolResult(Frame):
    """Result of a tool execution (matches OpenAI ChatCompletionToolMessageParam)."""

    call_id: str
    content: str

    @classmethod
    def new(cls, *, call_id: str, content: str) -> ToolResult:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            call_id=call_id,
            content=content,
        )


# Body tracking point names matching OpenVR full-body tracking
BODY_PARTS = (
    "head",
    "left_hand",
    "right_hand",
    "waist",
    "chest",
    "left_foot",
    "right_foot",
    "left_knee",
    "right_knee",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
)


class BodyPoseFrame(Frame):
    """Frame containing full-body pose data (positions + quaternion rotations).

    Each body part is a BonePose(pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z).
    Any body part can be None to indicate it should not be updated.
    """

    _poses: dict[str, BonePose | None]

    def __init__(
        self,
        display_name: str = "body_pose",
        *,
        poses: dict[str, BonePose | None],
        pts: int | None = None,
        id: int | None = None,
    ):
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "pts", pts if pts is not None else time.time_ns())
        object.__setattr__(self, "id", id if id is not None else obj_id())
        object.__setattr__(self, "_poses", poses)

    def get(self) -> dict[str, BonePose | None]:
        """Returns dict mapping body part name to BonePose (or None)."""
        return self._poses

    def __str__(self):
        active = sum(1 for v in self._poses.values() if v is not None)
        return f"BodyPoseFrame(id={self.id}, active_parts={active}/{len(self._poses)}, pts={self.pts})"


@dataclass(frozen=True, slots=True)
class ObjectDetectionFrame(Frame):
    """Object detection results for a single video frame."""

    boxes: np.ndarray
    """(N, M, 4) float32 — XYXY bounding boxes. N = objects, M = max # of detection per object."""

    scores: np.ndarray
    """(N, M) float32 — detection confidence. 0 means empty slot."""

    prompts: tuple[str, ...]
    """Objects to track, corresponding to each row in boxes/scores. N = len(prompts)"""

    @classmethod
    def new(
        cls,
        *,
        boxes: np.ndarray,
        scores: np.ndarray,
        prompts: tuple[str, ...],
    ) -> ObjectDetectionFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            boxes=boxes,
            scores=scores,
            prompts=prompts,
        )


@dataclass(frozen=True, slots=True)
class ObjectSegmentationFrame(Frame):
    """Instance segmentation results for a single video frame."""

    masks: np.ndarray
    """(K, H, W) bool — per-instance binary masks."""

    boxes: np.ndarray
    """(K, 4) float32 — XYXY bounding boxes."""

    scores: np.ndarray
    """(K,) float32 — detection confidence."""

    object_ids: np.ndarray
    """(K,) int64 — SAM3 tracking object IDs."""

    labels: tuple[str, ...]
    """(K,) — prompt label for each detection."""

    @classmethod
    def new(
        cls,
        *,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        object_ids: np.ndarray,
        labels: tuple[str, ...],
    ) -> ObjectSegmentationFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            masks=masks,
            boxes=boxes,
            scores=scores,
            object_ids=object_ids,
            labels=labels,
        )


@dataclass(frozen=True, slots=True)
class ObjectLocationFrame(Frame):
    """Per-object 3D world locations derived from segmentation + depth."""

    labels: tuple[str, ...]
    """(K,) — label for each detected object."""

    positions: np.ndarray
    """(K, 3) float32 — world-frame (x, y, z) position for each object."""

    depths: np.ndarray
    """(K,) float32 — median depth in metres for each object."""

    scores: np.ndarray
    """(K,) float32 — segmentation confidence score for each object."""

    boxes: np.ndarray
    """(K, 4) float32 — XYXY bounding boxes in the segmentation coordinate frame."""

    object_ids: np.ndarray
    """(K,) int64 — tracking object IDs from the segmenter."""

    @classmethod
    def new(
        cls,
        *,
        labels: tuple[str, ...],
        positions: np.ndarray,
        depths: np.ndarray,
        scores: np.ndarray,
        boxes: np.ndarray,
        object_ids: np.ndarray,
    ) -> ObjectLocationFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            labels=labels,
            positions=positions,
            depths=depths,
            scores=scores,
            boxes=boxes,
            object_ids=object_ids,
        )


@dataclass(frozen=True, slots=True)
class GoalFrame(Frame):
    """Frame containing a 3D goal coordinate for motion control."""

    x: float
    y: float
    z: float = 0.0

    @classmethod
    def new(cls, *, x: float, y: float, z: float = 0.0) -> GoalFrame:
        return cls(pts=time.time_ns(), id=obj_id(), x=x, y=y, z=z)


@dataclass(frozen=True, slots=True)
class CameraParamsFrame(Frame):
    """Camera intrinsics (3x3) and extrinsics (4x4) for a single frame."""

    intrinsics: np.ndarray
    extrinsics: np.ndarray
    width: int
    height: int

    @classmethod
    def new(
        cls,
        *,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        width: int,
        height: int,
    ) -> CameraParamsFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            width=width,
            height=height,
        )


@dataclass(frozen=True, slots=True)
class DepthFrame(Frame):
    """Per-pixel depth map (HxW float32)."""

    data: np.ndarray
    width: int
    height: int
    is_metric: bool

    @classmethod
    def new(cls, *, data: np.ndarray, is_metric: bool = False) -> DepthFrame:
        h, w = data.shape[:2]
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            data=data,
            width=w,
            height=h,
            is_metric=is_metric,
        )


class VideoDataFormat(Enum):
    BGR = "bgr"
    RGB = "rgb"


@dataclass(frozen=True, slots=True)
class VideoFrame(Frame):
    """Video frame carrying raw pixel data (always ndarray).

    Encoding (JPEG/PNG) is a boundary concern handled by sinks, not by the frame.
    """

    data: np.ndarray
    width: int
    height: int
    format: VideoDataFormat

    @classmethod
    def new(
        cls,
        *,
        data: np.ndarray,
        format: VideoDataFormat = VideoDataFormat.BGR,
    ) -> VideoFrame:
        h, w = data.shape[:2]
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            data=data,
            width=w,
            height=h,
            format=format,
        )

    def get(self, format: VideoDataFormat) -> np.ndarray:
        """Get pixel data in the requested color format."""
        if self.format == format:
            return self.data
        import cv2

        conv = {
            (VideoDataFormat.BGR, VideoDataFormat.RGB): cv2.COLOR_BGR2RGB,
            (VideoDataFormat.RGB, VideoDataFormat.BGR): cv2.COLOR_RGB2BGR,
        }
        return cv2.cvtColor(self.data, conv[(self.format, format)])


@dataclass(frozen=True, slots=True)
class StereoVideoFrame(Frame):
    """Stereo video frame carrying left and right eye pixel data."""

    left: np.ndarray
    right: np.ndarray
    width: int
    height: int
    format: VideoDataFormat

    @classmethod
    def new(
        cls,
        *,
        left: np.ndarray,
        right: np.ndarray,
        format: VideoDataFormat = VideoDataFormat.BGR,
    ) -> StereoVideoFrame:
        h, w = left.shape[:2]
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            left=left,
            right=right,
            width=w,
            height=h,
            format=format,
        )

    def get(self, eye: Literal["left", "right"], format: VideoDataFormat) -> np.ndarray:
        """Get pixel data for the requested eye in the requested color format."""
        data = self.left if eye == "left" else self.right
        if self.format == format:
            return data
        import cv2

        conv = {
            (VideoDataFormat.BGR, VideoDataFormat.RGB): cv2.COLOR_BGR2RGB,
            (VideoDataFormat.RGB, VideoDataFormat.BGR): cv2.COLOR_RGB2BGR,
        }
        return cv2.cvtColor(data, conv[(self.format, format)])


@dataclass(frozen=True, slots=True)
class StereoCameraParamsFrame(Frame):
    """Stereo camera parameters: intrinsics (3x3), extrinsics (4x4), and baseline (metres)."""

    intrinsics: np.ndarray
    extrinsics: np.ndarray
    baseline: float
    width: int
    height: int

    @classmethod
    def new(
        cls,
        *,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        baseline: float,
        width: int,
        height: int,
    ) -> StereoCameraParamsFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            baseline=baseline,
            width=width,
            height=height,
        )
