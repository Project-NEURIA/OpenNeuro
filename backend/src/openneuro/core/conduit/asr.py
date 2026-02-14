from __future__ import annotations

import os
import tempfile
import threading
import traceback
import wave
from queue import Queue, Empty
from typing import Any

import numpy as np
import requests

from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import AudioFrame, InterruptFrame, TextFrame


class ASR(Component[Channel[AudioFrame]]):
    """Automatic Speech Recognition component using Groq's Whisper API.
    
    This component processes audio segments and converts them to text transcriptions.
    It handles interrupt frames and speech segments, performing ASR on complete
    speech segments using the Groq Whisper API.
    """

    def __init__(
        self,
        *,
        groq_api_key: str | None = None,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
        timeout: int = 60,
    ) -> None:
        super().__init__()
        self._output = Channel[AudioFrame]()

        # API configuration
        self._api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError("GROQ_API_KEY must be provided either as parameter or environment variable")
        
        self._url = "https://api.groq.com/openai/v1/audio/transcriptions"
        self._model = model
        self._language = language
        self._timeout = timeout

        # Processing queues
        self._task_queue: Queue[AudioFrame] = Queue()
        self._worker_thread: threading.Thread | None = None

    def get_output_channels(self) -> tuple[Channel[AudioFrame]]:
        return (self._output,)

    def set_input_channels(self, t1: Channel[AudioFrame]) -> None:
        self._input_channel = t1

    def _prepare_audio_for_transcription(self, frame: AudioFrame) -> str:
        """Convert audio frame to temporary WAV file for transcription."""
        # Convert to mono 16kHz if needed
        pcm_data = np.frombuffer(frame.pcm16_data, dtype=np.int16)
        
        # Convert stereo to mono if needed
        if frame.channels == 2:
            pcm_mono = pcm_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        else:
            pcm_mono = pcm_data
        
        # Resample to 16kHz for Whisper
        if frame.sample_rate != 16000:
            # Simple linear interpolation for resampling
            resample_ratio = 16000 / frame.sample_rate
            target_length = int(len(pcm_mono) * resample_ratio)
            indices = np.linspace(0, len(pcm_mono) - 1, target_length)
            pcm_16k = np.interp(indices, np.arange(len(pcm_mono)), pcm_mono).astype(np.int16)
        else:
            pcm_16k = pcm_mono

        # Create temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(pcm_16k.tobytes())
            
            return temp_path
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def _transcribe_audio(self, frame: AudioFrame) -> TextFrame | None:
        """Transcribe audio frame using Groq Whisper API."""
        try:
            # Prepare audio file
            wav_path = self._prepare_audio_for_transcription(frame)
            
            try:
                # Make API request
                headers = {"Authorization": f"Bearer {self._api_key}"}
                data = {
                    "model": self._model,
                    "language": self._language,
                    "response_format": "json"
                }
                
                with open(wav_path, "rb") as audio_file:
                    files = {"file": ("audio.wav", audio_file, "audio/wav")}
                    response = requests.post(
                        self._url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=self._timeout
                    )
                
                response.raise_for_status()
                result = response.json()
                
                text = result.get("text", "").strip()
                if text:
                    print(f"[ASR] Transcription: '{text}'")
                    return TextFrame(
                        frame_type_string="asr_transcription",
                        text=text,
                        language=self._language,
                        pts=frame.pts
                    )
                else:
                    print("[ASR] No speech detected in audio segment")
                    return None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                    
        except requests.exceptions.RequestException as e:
            print(f"[ASR] API request failed: {e}")
            return None
        except Exception as e:
            print(f"[ASR] Transcription error: {e}")
            return None

    def _worker_loop(self) -> None:
        """Worker thread for processing transcription tasks."""
        print("[ASR] Worker thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get task from queue with timeout
                frame = self._task_queue.get(timeout=0.1)
                
                # Process transcription
                text_frame = self._transcribe_audio(frame)
                if text_frame:
                    self._output.send(text_frame)
                    
            except Exception as e:
                # Check if it's just an empty queue exception
                if isinstance(e, Empty):
                    continue  # Normal timeout, just continue
                elif not self.stop_event.is_set():
                    print(f"[ASR] Worker error: {e}")
                    traceback.print_exc()
                continue
        
        print("[ASR] Worker thread stopped")

    def run(self) -> None:
        """Main processing loop."""
        print("[ASR] Starting Automatic Speech Recognition")
        
        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        try:
            for frame in self._input_channel.stream(self.stop_event):
                if frame is None:
                    break
                
                # Handle interrupt frames
                if isinstance(frame, InterruptFrame):
                    print(f"[ASR] Passing through interrupt: {frame.reason}")
                    self._output.send(frame)
                    continue
                
                # Handle speech segments for transcription
                if frame.frame_type_string == "vad_speech_segment":
                    print(f"[ASR] Queuing speech segment for transcription: {len(frame.pcm16_data)} bytes")
                    self._task_queue.put(frame)
                else:
                    # Pass through other frames
                    self._output.send(frame)
                    
        finally:
            # Wait for worker thread to finish
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
        
        print("[ASR] Automatic Speech Recognition stopped")
