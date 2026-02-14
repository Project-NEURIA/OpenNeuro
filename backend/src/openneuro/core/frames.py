from __future__ import annotations

import sys
import threading
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict
import time
import numpy as np
import librosa

from openneuro.core.utils import obj_id


# Global frame registry - keeps last 100 frames
_MAX_FRAMES = 100


class FrameRegistry:
    """Singleton registry for tracking all frames with 100-frame limit."""

    _instance: FrameRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> FrameRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._frames: OrderedDict[int, Frame] = OrderedDict()
                    cls._instance._registry_lock = threading.Lock()
        return cls._instance

    def register(self, frame: Frame) -> None:
        """Register a frame, maintaining 100-frame limit (FIFO eviction)."""
        with self._registry_lock:
            self._frames[frame.id] = frame
            # Move to end to maintain insertion order
            self._frames.move_to_end(frame.id)
            # Evict oldest if over limit
            while len(self._frames) > _MAX_FRAMES:
                self._frames.popitem(last=False)

    def get_all(self) -> list[Frame]:
        """Return all frames sorted by PTS descending (newest first)."""
        with self._registry_lock:
            return sorted(self._frames.values(), key=lambda f: f.pts, reverse=True)

    def clear(self) -> None:
        """Clear all frames from registry."""
        with self._registry_lock:
            self._frames.clear()


# Global registry instance
_frame_registry = FrameRegistry()


@dataclass
class Frame(ABC):
    """Base frame class for all frames in the pipeline.

    Parameters:
        frame_type_string: String representing the type of the frame.
        pts: Presentation timestamp in nanoseconds.
        id: Unique identifier for the frame instance.
    """

    frame_type_string: str
    pts: int = field(default_factory=time.time_ns)
    id: int = field(default_factory=obj_id)

    def __post_init__(self):
        """Auto-register frame in global registry."""
        _frame_registry.register(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to dictionary representation."""
        return {
            "frame_type_string": self.frame_type_string,
            "pts": self.pts,
            "id": self.id,
            "message": str(self),  # Add string representation for debugging
        }

    def size_bytes(self) -> int:
        """Return approximate size of frame in bytes."""
        return sys.getsizeof(self)

    def __str__(self):
        return f"Frame(id={self.id}, type={self.frame_type_string}, pts={self.pts})"


class AudioFrame(Frame):
    """Audio frame containing PCM16 data with resampling capabilities."""

    def __init__(
        self,
        frame_type_string: str = "audio",
        *,
        pcm16_data: bytes,
        sample_rate: int,
        channels: int = 1,
        pts: int | None = None,
        id: int | None = None,
    ):
        pts = pts if pts is not None else time.time_ns()
        fid = id if id is not None else obj_id()
        super().__init__(frame_type_string=frame_type_string, pts=pts, id=fid)
        self.pcm16_data = pcm16_data
        self.sample_rate = sample_rate
        self.channels = channels

    def to_dict(self) -> Dict[str, Any]:
        """Convert audio frame to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "pcm16_data": self.pcm16_data,
            "sample_rate": self.sample_rate,
            "sr": self.sample_rate,  # Add sr alias for sample rate
            "channels": self.channels,
        })
        return base_dict

    def resample(self, target_sample_rate: int) -> AudioFrame:
        """Resample audio data to target sample rate."""
        if self.sample_rate == target_sample_rate:
            return AudioFrame(
                frame_type_string=self.frame_type_string,
                pcm16_data=self.pcm16_data,
                sample_rate=self.sample_rate,
                channels=self.channels,
                pts=self.pts,
                id=self.id,
            )

        # Convert bytes to numpy array
        audio_data = np.frombuffer(self.pcm16_data, dtype=np.int16)
        
        # Reshape for stereo if needed
        if self.channels == 2:
            audio_data = audio_data.reshape(-1, 2)
        else:
            audio_data = audio_data.reshape(-1, 1)

        # Resample using librosa
        if self.channels == 2:
            # Resample each channel separately
            resampled_channels = []
            for ch in range(self.channels):
                channel_data = audio_data[:, ch]
                resampled = librosa.resample(
                    channel_data.astype(np.float32) / 32768.0,
                    orig_sr=self.sample_rate,
                    target_sr=target_sample_rate
                )
                resampled_channels.append((resampled * 32767.0).astype(np.int16))
            resampled_audio = np.column_stack(resampled_channels)
        else:
            # Mono resampling
            resampled_float = librosa.resample(
                audio_data.flatten().astype(np.float32) / 32768.0,
                orig_sr=self.sample_rate,
                target_sr=target_sample_rate
            )
            resampled_audio = (resampled_float * 32767.0).astype(np.int16).reshape(-1, 1)

        return AudioFrame(
            frame_type_string=self.frame_type_string,
            pcm16_data=resampled_audio.tobytes(),
            sample_rate=target_sample_rate,
            channels=self.channels,
            pts=self.pts,
            id=self.id,
        )

    def __str__(self):
        duration_ms = len(self.pcm16_data) / (self.sample_rate * self.channels * 2) * 1000
        return f"AudioFrame(id={self.id}, type={self.frame_type_string}, pts={self.pts}, duration={duration_ms:.1f}ms, sr={self.sample_rate}Hz)"


class InterruptFrame(Frame):
    """Frame representing an interrupt event in the audio pipeline."""

    def __init__(
        self,
        frame_type_string: str = "interrupt",
        *,
        reason: str = "speech_detected",
        pts: int | None = None,
        id: int | None = None,
    ):
        pts = pts if pts is not None else time.time_ns()
        fid = id if id is not None else obj_id()
        super().__init__(frame_type_string=frame_type_string, pts=pts, id=fid)
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert interrupt frame to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "reason": self.reason,
        })
        return base_dict

    def __str__(self):
        return f"InterruptFrame(id={self.id}, reason={self.reason}, pts={self.pts})"


class TextFrame(Frame):
    """Frame containing text data such as transcriptions or responses."""

    def __init__(
        self,
        frame_type_string: str = "text",
        *,
        text: str,
        language: str | None = None,
        pts: int | None = None,
        id: int | None = None,
    ):
        pts = pts if pts is not None else time.time_ns()
        fid = id if id is not None else obj_id()
        super().__init__(frame_type_string=frame_type_string, pts=pts, id=fid)
        self.text = text
        self.language = language

    def to_dict(self) -> Dict[str, Any]:
        """Convert text frame to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "text": self.text,
            "language": self.language,
        })
        return base_dict

    def __str__(self):
        text_preview = self.text
        return f"TextFrame(id={self.id}, text='{text_preview}', pts={self.pts})"


class ConversationalFrame(TextFrame):
    """Frame containing conversational context with messages for LLM APIs.
    
    Wraps TextFrame and adds messages parameter for chat completions APIs.
    """

    def __init__(
        self,
        frame_type_string: str = "conversational",
        *,
        text: str,
        messages: list[dict[str, str]],
        language: str | None = None,
        pts: int | None = None,
        id: int | None = None,
    ):
        pts = pts if pts is not None else time.time_ns()
        fid = id if id is not None else obj_id()
        super().__init__(frame_type_string=frame_type_string, text=text, language=language, pts=pts, id=fid)
        self.messages = messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversational frame to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "messages": self.messages,
        })
        return base_dict

    def __str__(self):
        text_preview = self.text
        return f"ConversationalFrame(id={self.id}, text='{text_preview}', messages={len(self.messages)}, pts={self.pts})"


def get_frame_registry() -> FrameRegistry:
    """Get the global frame registry instance."""
    return _frame_registry
