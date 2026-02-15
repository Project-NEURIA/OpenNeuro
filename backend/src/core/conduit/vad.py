from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import TypedDict

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperFeatureExtractor

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import AudioFrame, InterruptFrame, AudioDataFormat
from src.core.config import BaseConfig


class VADConfig(BaseConfig):
    silence_seconds: float = 0.9
    max_silence_seconds: float = 1.4
    pre_speech_seconds: float = 1.0
    min_speech_seconds: float = 0.5
    turn_threshold: float = 0.89
    # assets/smart-turn-v3.0.onnx relative to project root
    smart_turn_onnx: str = "assets/smart-turn-v3.0.onnx"


class VADOutputs(TypedDict):
    audio: Channel[AudioFrame]
    interrupt: Channel[InterruptFrame]


class VAD(Component[[Channel[AudioFrame]], VADOutputs]):
    def __init__(self, config: VADConfig | None = None) -> None:
        super().__init__(config or VADConfig())
        self.config: VADConfig

        self._output_audio = Channel[AudioFrame](name="audio")
        self._output_interrupt = Channel[InterruptFrame](name="interrupt")

        # Load models
        self._load_silero_vad()
        self._load_smart_turn_model()

        # State tracking
        self._speaking = False
        self._silence_start: float | None = None

        # Buffers for turn detection and finalization
        self._pre_buffer: list[AudioFrame] = []
        self._current_segment: list[AudioFrame] = []

        # Buffer for Silero VAD (always 16kHz mono)
        self._vad_buffer: deque[float] = deque(maxlen=4000)
        self._lock = threading.Lock()

    def get_output_channels(self) -> VADOutputs:
        return {
            "audio": self._output_audio,
            "interrupt": self._output_interrupt,
        }

    def _load_silero_vad(self) -> None:
        self._silero_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _, _, _, self._VADIterator, _ = utils
        self._vad_iterator = self._VADIterator(self._silero_model)

    def _load_smart_turn_model(self) -> None:
        onnx_path = self.config.smart_turn_onnx
        if not os.path.exists(onnx_path):
            print(
                f"[VAD] Smart Turn model not found at {onnx_path}, disabling smart turn detection"
            )
            self._smart_turn_session = None
            return

        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1

        self._smart_turn_session = ort.InferenceSession(
            onnx_path, sess_options=session_options
        )
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        print(f"[VAD] Smart Turn model loaded from {onnx_path}")

    def _check_smart_turn(self) -> bool:
        if self._smart_turn_session is None or not self._current_segment:
            return False

        try:
            # For smart turn check, we still need a 16kHz mono version of the current segment
            # We can get it by sampling the frames in _current_segment
            segment_data = []
            for f in self._current_segment:
                segment_data.append(
                    f.get(
                        sample_rate=16000,
                        num_channels=1,
                        data_format=AudioDataFormat.FLOAT32,
                    ).flatten()
                )

            pcm_16k = np.concatenate(segment_data)

            max_samples = 8 * 16000
            if len(pcm_16k) > max_samples:
                pcm_16k = pcm_16k[-max_samples:]
            else:
                pcm_16k = np.pad(
                    pcm_16k, (max_samples - len(pcm_16k), 0), mode="constant"
                )

            inputs = self._feature_extractor(
                pcm_16k,
                sampling_rate=16000,
                return_tensors="np",
                padding="max_length",
                max_length=max_samples,
                truncation=True,
                do_normalize=True,
            )

            input_features = inputs.input_features.squeeze(0).astype(np.float32)
            input_features = np.expand_dims(input_features, axis=0)

            outputs = self._smart_turn_session.run(
                None, {"input_features": input_features}
            )
            turn_probability = outputs[0][0].item()

            return turn_probability > self.config.turn_threshold

        except Exception as e:
            print(f"[VAD] Error in Smart Turn detection: {e}")
            return False

    def _process_audio_frame(self, frame: AudioFrame) -> None:
        with self._lock:
            # 1. Prepare data for VAD (16kHz mono)
            pcm_16k = frame.get(
                sample_rate=16000, num_channels=1, data_format=AudioDataFormat.FLOAT32
            ).flatten()
            self._vad_buffer.extend(pcm_16k.tolist())

            # 2. Run VAD loop
            while len(self._vad_buffer) >= 512:
                chunk = torch.tensor(list(self._vad_buffer)[:512])
                for _ in range(512):
                    self._vad_buffer.popleft()

                vad_result = self._vad_iterator(chunk, return_seconds=False)

                if vad_result and "start" in vad_result and not self._speaking:
                    self._handle_speech_start()

                if vad_result and "end" in vad_result and self._speaking:
                    if self._silence_start is None:
                        self._silence_start = time.time()

            # 3. Handle segment buffering
            if self._speaking:
                self._current_segment.append(frame)
                if self._silence_start is not None:
                    silence_duration = time.time() - self._silence_start
                    if silence_duration >= self.config.max_silence_seconds:
                        print(f"[VAD] Max silence reached: {silence_duration:.2f}s")
                        self._finalize_segment()
                    elif silence_duration >= self.config.silence_seconds:
                        if self._check_smart_turn():
                            print(
                                f"[VAD] Smart Turn detected after silence: {silence_duration:.2f}s"
                            )
                            self._finalize_segment()
            else:
                self._pre_buffer.append(frame)
                # Keep pre-buffer within limits (seconds based)
                total_ms = sum(
                    f._data.shape[1] / f._sample_rate * 1000 for f in self._pre_buffer
                )
                while (
                    total_ms > (self.config.pre_speech_seconds * 1000)
                    and self._pre_buffer
                ):
                    f_removed = self._pre_buffer.pop(0)
                    total_ms -= f_removed._data.shape[1] / f_removed._sample_rate * 1000

    def _handle_speech_start(self) -> None:
        print("[VAD] Speech started")
        self._speaking = True
        self._silence_start = None

        self._output_interrupt.send(
            InterruptFrame(display_name="vad_interrupt", reason="speech_detected")
        )

        self._current_segment = list(self._pre_buffer)
        self._pre_buffer = []

    def _finalize_segment(self) -> None:
        if not self._current_segment:
            return

        # Defer resampling to the end: concatenate internal float32 data from all frames.
        # We assume all frames in the segment have the same sample rate and channels.
        first_frame = self._current_segment[0]
        sr = first_frame._sample_rate
        ch = first_frame._channels

        segment_data_list = []
        for f in self._current_segment:
            # We use get(FLOAT32) without SR/CH change to get the "raw" normalized float32 data
            segment_data_list.append(f.get(data_format=AudioDataFormat.FLOAT32))

        all_data = np.concatenate(segment_data_list, axis=1)

        self._current_segment = []
        self._speaking = False
        self._silence_start = None
        self._vad_iterator.reset_states()

        duration = all_data.shape[1] / sr
        if duration >= self.config.min_speech_seconds:
            output_frame = AudioFrame(
                display_name="vad_speech_segment",
                data=all_data,
                sample_rate=sr,
                channels=ch,
            )
            self._output_audio.send(output_frame)
            print(
                f"[VAD] Speech segment finalized: {duration:.2f}s ({all_data.shape[1]} samples at {sr}Hz)"
            )
        else:
            print(f"[VAD] Segment too short: {duration:.2f}s")

    def _monitor_loop(self) -> None:
        """Background thread to finalize segments if the source is silent."""
        while not self.stop_event.is_set():
            time.sleep(0.1)
            with self._lock:
                if self._speaking and self._silence_start is not None:
                    silence_duration = time.time() - self._silence_start
                    if silence_duration >= self.config.max_silence_seconds:
                        print(
                            f"[VAD] Monitor: Max silence reached ({silence_duration:.2f}s)"
                        )
                        self._finalize_segment()
                    elif silence_duration >= self.config.silence_seconds:
                        if self._check_smart_turn():
                            print(
                                f"[VAD] Monitor: Smart Turn detected after silence ({silence_duration:.2f}s)"
                            )
                            self._finalize_segment()

    def run(self, audio: Channel[AudioFrame] | None = None) -> None:
        print("[VAD] Starting Voice Activity Detection")

        # Start proactive silence monitor
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()

        if audio:
            for frame in audio.stream(self):
                if frame is None:
                    break
                self._process_audio_frame(frame)

        if self._current_segment:
            with self._lock:
                self._finalize_segment()

        print("[VAD] Voice Activity Detection stopped")
