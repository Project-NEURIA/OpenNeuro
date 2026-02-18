from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.core.utils import obj_id


class AudioDataFormat(Enum):
    PCM8 = "pcm8"
    PCM16 = "pcm16"
    FLOAT32 = "float32"


class MessagesDataFormat(Enum):
    MESSAGES = "messages"
    TEXT = "text"


@dataclass(frozen=True)
class Frame(ABC):
    """Base frame class for all frames in the pipeline."""

    display_name: str
    pts: int = field(default_factory=time.time_ns)
    id: int = field(default_factory=obj_id)

    @abstractmethod
    def get(self, *args, **kwargs) -> Any:
        """Get the data from the frame in a specific format."""
        pass

    def __str__(self):
        return f"{type(self).__name__}(id={self.id}, type={self.display_name}, pts={self.pts})"


class AudioFrame(Frame):
    """Audio frame with immutable data and on-the-fly reformatting/resampling."""

    _data: np.ndarray
    _sample_rate: int
    _channels: int

    def __init__(
        self,
        display_name: str = "audio",
        *,
        data: bytes | np.ndarray,
        sample_rate: int,
        channels: int = 1,
        pts: int | None = None,
        id: int | None = None,
    ):
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "pts", pts if pts is not None else time.time_ns())
        object.__setattr__(self, "id", id if id is not None else obj_id())

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

        object.__setattr__(self, "_data", arr)
        object.__setattr__(self, "_sample_rate", sample_rate)
        object.__setattr__(self, "_channels", channels)

    def get(
        self,
        sample_rate: int | None = None,
        num_channels: int | None = None,
        data_format: AudioDataFormat = AudioDataFormat.PCM16,
    ) -> Any:
        """Get the audio data in the requested format, sample rate, and channels."""
        arr = self._data
        current_sr = self._sample_rate
        current_ch = self._channels

        # 1. Resample if needed
        if sample_rate and sample_rate != current_sr:
            num_samples = int(arr.shape[1] * sample_rate / current_sr)
            # Linear interpolation for resampling
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

    def __str__(self):
        duration_ms = self._data.shape[1] / self._sample_rate * 1000
        return f"AudioFrame(id={self.id}, pts={self.pts}, duration={duration_ms:.1f}ms, sr={self._sample_rate}Hz, channels={self._channels})"


class TextFrame(Frame):
    """Frame containing text data."""

    _text: str

    def __init__(
        self,
        display_name: str = "text",
        *,
        text: str,
        language: str | None = None,
        pts: int | None = None,
        id: int | None = None,
    ):
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "pts", pts if pts is not None else time.time_ns())
        object.__setattr__(self, "id", id if id is not None else obj_id())
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_language", language)

    def get(self) -> str:
        return self._text

    def __str__(self):
        return f"TextFrame(id={self.id}, text='{self._text[:50]}...', pts={self.pts})"


class InterruptFrame(Frame):
    """Frame representing an interrupt event."""

    _reason: str

    def __init__(
        self,
        display_name: str = "interrupt",
        *,
        reason: str = "speech_detected",
        pts: int | None = None,
        id: int | None = None,
    ):
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "pts", pts if pts is not None else time.time_ns())
        object.__setattr__(self, "id", id if id is not None else obj_id())
        object.__setattr__(self, "_reason", reason)

    def get(self) -> str:
        return self._reason

    def __str__(self):
        return f"InterruptFrame(id={self.id}, reason={self._reason}, pts={self.pts})"


class MessagesFrame(Frame):
    """Frame containing conversation history (messages) for LLM consumption."""

    _text: str
    _messages: list[dict[str, str]]

    def __init__(
        self,
        display_name: str = "messages",
        *,
        text: str,
        messages: list[dict[str, str]],
        language: str | None = None,
        pts: int | None = None,
        id: int | None = None,
    ):
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "pts", pts if pts is not None else time.time_ns())
        object.__setattr__(self, "id", id if id is not None else obj_id())
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_messages", messages)
        object.__setattr__(self, "_language", language)

    def get(self, format: MessagesDataFormat = MessagesDataFormat.MESSAGES) -> Any:
        if format == MessagesDataFormat.MESSAGES:
            return self._messages
        if format == MessagesDataFormat.TEXT:
            return self._text
        raise ValueError(f"Unsupported format: {format}")

    def __str__(self):
        return f"MessagesFrame(id={self.id}, msg_count={len(self._messages)}, pts={self.pts})"
