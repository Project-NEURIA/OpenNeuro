from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, overload, NamedTuple

import numpy as np

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
class MessagesFrame(Frame):
    """Frame containing conversation history (messages) for LLM consumption."""

    text: str
    messages: list[dict[str, str]]
    language: str | None = None

    @classmethod
    def new(
        cls,
        *,
        text: str,
        messages: list[dict[str, str]],
        language: str | None = None,
    ) -> MessagesFrame:
        return cls(
            pts=time.time_ns(),
            id=obj_id(),
            text=text,
            messages=messages,
            language=language,
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
