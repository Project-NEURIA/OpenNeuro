from __future__ import annotations

import os
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import TypedDict

import requests

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import AudioFrame, InterruptFrame, TextFrame, AudioDataFormat
from src.core.config import BaseConfig


class ASRConfig(BaseConfig):
    groq_api_key: str | None = None
    model: str = "whisper-large-v3-turbo"
    language: str = "en"
    timeout: int = 60


class ASROutputs(TypedDict):
    text: Channel[TextFrame]
    interrupt: Channel[InterruptFrame]


class ASR(Component[[Channel[AudioFrame], Channel[InterruptFrame]], ASROutputs]):
    def __init__(self, config: ASRConfig | None = None) -> None:
        super().__init__(config or ASRConfig())
        self.config: ASRConfig
        self._output_text = Channel[TextFrame](name="text")
        self._output_interrupt = Channel[InterruptFrame](name="interrupt")

        self._api_key = self.config.groq_api_key or os.getenv("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY must be provided either as parameter or environment variable"
            )

        self._url = "https://api.groq.com/openai/v1/audio/transcriptions"

        self._task_queue: Queue[AudioFrame] = Queue()
        self._worker_thread: threading.Thread | None = None

    def get_output_channels(self) -> ASROutputs:
        return {
            "text": self._output_text,
            "interrupt": self._output_interrupt,
        }

    def _prepare_audio_for_transcription(self, frame: AudioFrame) -> str:
        # Whisper prefers 16kHz mono PCM16
        pcm_16k = frame.get(
            sample_rate=16000, num_channels=1, data_format=AudioDataFormat.PCM16
        )

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_16k)
            return temp_path
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def _save_debug_audio(self, wav_path: str) -> None:
        """Save a copy of the audio file to the debug directory."""
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_path = debug_dir / f"groq_audio_{timestamp}.wav"

        try:
            with open(wav_path, "rb") as src, open(debug_path, "wb") as dst:
                dst.write(src.read())
            print(f"[ASR] Debug audio saved to: {debug_path}")
        except Exception as e:
            print(f"[ASR] Failed to save debug audio: {e}")

    def _transcribe_audio(self, frame: AudioFrame) -> TextFrame | None:
        try:
            wav_path = self._prepare_audio_for_transcription(frame)
            try:
                # # Save debug audio before sending to Groq
                # self._save_debug_audio(wav_path)

                headers = {"Authorization": f"Bearer {self._api_key}"}
                data = {
                    "model": self.config.model,
                    "language": self.config.language,
                    "response_format": "json",
                }

                with open(wav_path, "rb") as audio_file:
                    files = {"file": ("audio.wav", audio_file, "audio/wav")}
                    response = requests.post(
                        self._url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=self.config.timeout,
                    )

                response.raise_for_status()
                result = response.json()

                text = result.get("text", "").strip()
                if text:
                    print(f"[ASR] Transcription: '{text}'")
                    return TextFrame(
                        display_name="asr_transcription",
                        text=text,
                        language=self.config.language,
                        pts=frame.pts,
                    )
                return None
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
        except Exception as e:
            print(f"[ASR] Transcription error: {e}")
            return None

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self._task_queue.get(timeout=0.1)
                text_frame = self._transcribe_audio(frame)
                if text_frame:
                    self._output_text.send(text_frame)
            except Empty:
                continue
            except Exception as e:
                print(f"[ASR] Worker error: {e}")
                continue

    def run(
        self,
        audio: Channel[AudioFrame] | None = None,
        interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        print("[ASR] Starting Automatic Speech Recognition")
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        if interrupt:

            def passthrough():
                for frame in interrupt.stream(self):
                    if frame is None:
                        break
                    self._output_interrupt.send(frame)

            threading.Thread(target=passthrough, daemon=True).start()

        if audio:
            try:
                for frame in audio.stream(self):
                    if frame is None:
                        break

                    # Handle speech segments from VAD
                    if frame.display_name == "vad_speech_segment":
                        self._task_queue.put(frame)
            finally:
                pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        print("[ASR] Automatic Speech Recognition stopped")
