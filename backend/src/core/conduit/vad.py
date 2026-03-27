from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
import torch
from pydantic import BaseModel
from transformers import WhisperFeatureExtractor

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioDataFormat, AudioFrame, InterruptFrame


class VADConfig(BaseModel):
    silence_seconds: float = 0.9
    max_silence_seconds: float = 1.4
    pre_speech_seconds: float = 1.0
    min_speech_seconds: float = 0.5
    turn_threshold: float = 0.89
    # assets/smart-turn-v3.0.onnx relative to project root
    smart_turn_onnx: str = "assets/smart-turn-v3.0.onnx"


class VADInputs(NamedTuple):
    audio: Receiver[AudioFrame]


class VADOutputs(NamedTuple):
    audio: Sender[AudioFrame]
    interrupt: Sender[InterruptFrame] | None = None


class VAD(ThreadedComponent[VADInputs, VADOutputs]):
    tags = Tag(io={"conduit"}, functionality={"audio"})
    description = "Detects **voice activity** in an audio stream using *Silero VAD* and *Smart Turn* models. Segments speech from silence with configurable thresholds, emitting trimmed audio and optional `InterruptFrame` signals."

    def __init__(self, config: VADConfig) -> None:
        super().__init__()
        self.config: VADConfig = config

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

    def _prepare_smart_turn_input(self) -> np.ndarray | None:
        """Copy segment data under lock for Smart Turn inference. Returns None if unavailable."""
        if self._smart_turn_session is None or not self._current_segment:
            return None

        try:
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
            return pcm_16k
        except Exception as e:
            print(f"[VAD] Error preparing Smart Turn input: {e}")
            return None

    def _run_smart_turn_inference(self, pcm_16k: np.ndarray) -> bool:
        """Run ONNX inference without holding the lock."""
        try:
            features = self._feature_extractor(
                pcm_16k,
                sampling_rate=16000,
                return_tensors="np",
                padding="max_length",
                max_length=8 * 16000,
                truncation=True,
                do_normalize=True,
            )

            input_features = features.input_features.squeeze(0).astype(np.float32)
            input_features = np.expand_dims(input_features, axis=0)

            results = self._smart_turn_session.run(
                None, {"input_features": input_features}
            )
            turn_probability = results[0][0].item()

            return turn_probability > self.config.turn_threshold

        except Exception as e:
            print(f"[VAD] Error in Smart Turn detection: {e}")
            return False

    def _process_audio_frame(self, frame: AudioFrame, outputs: VADOutputs) -> None:
        smart_turn_input: np.ndarray | None = None

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
                    self._handle_speech_start(outputs)

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
                        self._finalize_segment(outputs)
                    elif silence_duration >= self.config.silence_seconds:
                        smart_turn_input = self._prepare_smart_turn_input()
            else:
                self._pre_buffer.append(frame)
                # Keep pre-buffer within limits (seconds based)
                total_ms = sum(
                    f.data.shape[1] / f.sample_rate * 1000 for f in self._pre_buffer
                )
                while (
                    total_ms > (self.config.pre_speech_seconds * 1000)
                    and self._pre_buffer
                ):
                    f_removed = self._pre_buffer.pop(0)
                    total_ms -= f_removed.data.shape[1] / f_removed.sample_rate * 1000

        # Run ONNX inference outside the lock
        if smart_turn_input is not None:
            if self._run_smart_turn_inference(smart_turn_input):
                with self._lock:
                    # Re-check state: segment may have been finalized by monitor thread
                    if self._speaking and self._silence_start is not None:
                        silence_duration = time.time() - self._silence_start
                        print(
                            f"[VAD] Smart Turn detected after silence: {silence_duration:.2f}s"
                        )
                        self._finalize_segment(outputs)

    def _handle_speech_start(self, outputs: VADOutputs) -> None:
        print("[VAD] Speech started")
        self._speaking = True
        self._silence_start = None

        if outputs.interrupt is not None:
            outputs.interrupt.send(InterruptFrame.new(reason="speech_detected"))

        self._current_segment = list(self._pre_buffer)
        self._pre_buffer = []

    def _finalize_segment(self, outputs: VADOutputs) -> None:
        if not self._current_segment:
            return

        # Defer resampling to the end: concatenate internal float32 data from all frames.
        # We assume all frames in the segment have the same sample rate and channels.
        first_frame = self._current_segment[0]
        sr = first_frame.sample_rate
        ch = first_frame.channels

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
            output_frame = AudioFrame.new(
                data=all_data,
                sample_rate=sr,
                channels=ch,
            )
            outputs.audio.send(output_frame)
            print(
                f"[VAD] Speech segment finalized: {duration:.2f}s ({all_data.shape[1]} samples at {sr}Hz)"
            )
        else:
            print(f"[VAD] Segment too short: {duration:.2f}s")

    def _monitor_loop(self, outputs: VADOutputs) -> None:
        """Background thread to finalize segments if the source is silent."""
        while not self.stop_event.is_set():
            time.sleep(0.1)
            smart_turn_input: np.ndarray | None = None

            with self._lock:
                if self._speaking and self._silence_start is not None:
                    silence_duration = time.time() - self._silence_start
                    if silence_duration >= self.config.max_silence_seconds:
                        print(
                            f"[VAD] Monitor: Max silence reached ({silence_duration:.2f}s)"
                        )
                        self._finalize_segment(outputs)
                    elif silence_duration >= self.config.silence_seconds:
                        smart_turn_input = self._prepare_smart_turn_input()

            # Run ONNX inference outside the lock
            if smart_turn_input is not None:
                if self._run_smart_turn_inference(smart_turn_input):
                    with self._lock:
                        # Re-check state: segment may have been finalized already
                        if self._speaking and self._silence_start is not None:
                            silence_duration = time.time() - self._silence_start
                            print(
                                f"[VAD] Monitor: Smart Turn detected after silence ({silence_duration:.2f}s)"
                            )
                            self._finalize_segment(outputs)

    def run(self, inputs: VADInputs, outputs: VADOutputs) -> None:
        print("[VAD] Starting Voice Activity Detection")

        # Start proactive silence monitor
        monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(outputs,), daemon=True
        )
        monitor_thread.start()

        try:
            for frame in inputs.audio:
                if frame is None:
                    break
                self._process_audio_frame(frame, outputs)

            if self._current_segment:
                with self._lock:
                    self._finalize_segment(outputs)
        finally:
            monitor_thread.join(timeout=2.0)

        print("[VAD] Voice Activity Detection stopped")
