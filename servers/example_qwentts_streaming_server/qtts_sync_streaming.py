#!/usr/bin/env python3
import threading
from pathlib import Path
from typing import Generator, Tuple, Union, Optional
from dataclasses import dataclass

import numpy as np

from qwen3streaming import SimpleStreamingTTS


@dataclass
class QsyncConfig:
    pass


class QsyncStreamingTTS:

    def __init__(self, config: Optional[QsyncConfig] = None):
        self.config = config or QsyncConfig()

        # Initialize TTS
        print("[Qsync] Loading TTS model...")
        self.tts = SimpleStreamingTTS()
        self.sample_rate = self.tts.sample_rate

        print(f"[Qsync] Ready. TTS SR={self.sample_rate}Hz")

    def register_voice(
        self,
        name: str,
        audio: Union[str, Tuple[np.ndarray, int]],
        transcript: str,
        x_vector_only: bool = False,
    ) -> Path:
        """Register a voice for cloning. Delegates to TTS."""
        return self.tts.register_voice(name, audio, transcript, x_vector_only)

    def generate_streaming(
        self,
        text: str,
        voice_prompt_path: Union[str, Path],
        language: str = "English",
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], dict], None, None]:
        for audio_chunk, meta in self.tts.generate_streaming(text, voice_prompt_path, language):
            if stop_event is not None and stop_event.is_set():
                yield np.array([], dtype=np.float32), None, {"is_final": True, "cancelled": True}
                return

            if meta.get("is_final"):
                yield np.array([], dtype=np.float32), None, meta
                continue

            if len(audio_chunk) == 0:
                continue

            yield audio_chunk, None, meta


# -----------------------------
# Global singleton
# -----------------------------

_qsync_tts = None


def get_qsync_tts(config: Optional[QsyncConfig] = None) -> QsyncStreamingTTS:
    """Get or create the global QsyncStreamingTTS instance."""
    global _qsync_tts
    if _qsync_tts is None:
        _qsync_tts = QsyncStreamingTTS(config)
    return _qsync_tts
