from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperFeatureExtractor

from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import AudioFrame, InterruptFrame


class VAD(Component[Channel[AudioFrame]]):
    """Voice Activity Detection component using Silero VAD and Smart Turn detection.
    
    This component detects speech segments in audio frames and outputs complete
    speech segments when speech ends. It uses Silero VAD for real-time speech
    detection and a Smart Turn model to determine when a speaker has finished
    their turn.
    """

    def __init__(
        self,
        *,
        silence_seconds: float = 0.9,
        max_silence_seconds: float = 1.4,
        pre_speech_seconds: float = 1.0,
        min_speech_seconds: float = 0.5,
        turn_threshold: float = 0.89,
        smart_turn_onnx: str = str(Path(__file__).resolve().parents[4] / "assets" / "smart-turn-v3.0.onnx"),
    ) -> None:
        super().__init__()
        self._output = Channel[AudioFrame]()

        # Timing configuration
        self._silence_seconds = silence_seconds
        self._max_silence_seconds = max_silence_seconds
        self._turn_threshold = turn_threshold
        
        # Audio buffer configuration (assuming 48kHz, 16-bit, mono)
        self._sample_rate = 48000
        self._bytes_per_sample = 2
        self._min_speech_bytes = int(self._sample_rate * self._bytes_per_sample * min_speech_seconds)
        self._max_pre_bytes = int(self._sample_rate * self._bytes_per_sample * pre_speech_seconds)

        # State tracking
        self._speaking = False
        self._silence_start: float | None = None
        self._pre_buffer = bytearray()
        self._current_segment = bytearray()
        self._vad_buffer: deque[float] = deque(maxlen=1536)  # 32ms of 16kHz audio
        
        # Thread safety
        self._lock = threading.Lock()

        # Load models
        self._load_silero_vad()
        self._load_smart_turn_model(smart_turn_onnx)

    def get_output_channels(self) -> tuple[Channel[AudioFrame]]:
        return (self._output,)

    def set_input_channels(self, t1: Channel[AudioFrame]) -> None:
        self._input_channel = t1

    def _load_silero_vad(self) -> None:
        """Load Silero VAD model for real-time speech detection."""
        self._silero_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._get_speech_timestamps, _, _, self._VADIterator, _ = utils
        self._vad_iterator = self._VADIterator(self._silero_model)

    def _load_smart_turn_model(self, onnx_path: str) -> None:
        """Load Smart Turn ONNX model for turn-taking detection."""
        if not os.path.exists(onnx_path):
            print(f"[VAD] Smart Turn model not found at {onnx_path}, disabling smart turn detection")
            self._smart_turn_session = None
            return
            
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self._smart_turn_session = ort.InferenceSession(onnx_path, sess_options=session_options)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        print(f"[VAD] Smart Turn model loaded from {onnx_path}")

    def _check_smart_turn(self) -> bool:
        """Check if the current segment is likely a complete turn using Smart Turn model."""
        if self._smart_turn_session is None:
            return False
            
        if not self._current_segment:
            return False

        try:
            # Convert stereo to mono if needed
            pcm_data = np.frombuffer(self._current_segment, dtype=np.int16)
            if len(pcm_data) % 2 == 0:  # Assume stereo if even length
                pcm_mono = pcm_data.reshape(-1, 2).mean(axis=1)
            else:
                pcm_mono = pcm_data
                
            # Resample to 16kHz for Smart Turn model
            pcm_16k = pcm_mono[::3].astype(np.float32) / 32768.0
            
            # Pad or truncate to 8 seconds (128000 samples at 16kHz)
            max_samples = 8 * 16000
            if len(pcm_16k) > max_samples:
                pcm_16k = pcm_16k[-max_samples:]
            else:
                pcm_16k = np.pad(pcm_16k, (max_samples - len(pcm_16k), 0), mode='constant')

            # Extract features
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
            
            # Run inference
            outputs = self._smart_turn_session.run(None, {"input_features": input_features})
            turn_probability = outputs[0][0].item()
            
            print(f"[VAD] Smart Turn probability: {turn_probability:.3f}")
            return turn_probability > self._turn_threshold
            
        except Exception as e:
            print(f"[VAD] Error in Smart Turn detection: {e}")
            return False

    def _process_audio_frame(self, frame: AudioFrame) -> None:
        """Process a single audio frame for VAD."""
        with self._lock:
            # Resample to 48kHz if needed
            if frame.sample_rate != self._sample_rate:
                frame = frame.resample(self._sample_rate)
            
            # Convert to mono if stereo
            pcm_data = np.frombuffer(frame.pcm16_data, dtype=np.int16)
            if frame.channels == 2:
                pcm_mono = pcm_data.reshape(-1, 2).mean(axis=1)
            else:
                pcm_mono = pcm_data
            
            # Convert to 16kHz for Silero VAD
            pcm_16k = pcm_mono[::3].astype(np.float32) / 32768.0
            
            # Add to VAD buffer
            self._vad_buffer.extend(pcm_16k.tolist())
            
            # Process VAD in chunks of 512 samples (32ms at 16kHz)
            while len(self._vad_buffer) >= 512:
                chunk = torch.tensor(list(self._vad_buffer)[:512])
                for _ in range(512):
                    self._vad_buffer.popleft()
                
                vad_result = self._vad_iterator(chunk, return_seconds=False)
                
                speech_start = vad_result and "start" in vad_result
                speech_end = vad_result and "end" in vad_result
                
                if speech_start and not self._speaking:
                    self._handle_speech_start()
                
                if speech_end and self._speaking:
                    # Set silence start time when speech ends
                    if self._silence_start is None:
                        self._silence_start = time.time()
            
            # Store audio data
            frame_bytes = frame.pcm16_data
            if self._speaking:
                self._current_segment.extend(frame_bytes)
                
                # Check silence duration continuously while speaking
                if self._silence_start is not None:
                    silence_duration = time.time() - self._silence_start
                    print(f"[VAD] Silence duration: {silence_duration:.2f}s")
                    
                    # Check if we should finalize the segment
                    if silence_duration >= self._max_silence_seconds:
                        self._finalize_segment()
                    elif silence_duration >= self._silence_seconds:
                        if self._check_smart_turn():
                            self._finalize_segment()
            else:
                self._pre_buffer.extend(frame_bytes)
                if len(self._pre_buffer) > self._max_pre_bytes:
                    self._pre_buffer = self._pre_buffer[-self._max_pre_bytes:]

    def _handle_speech_start(self) -> None:
        """Handle the start of speech detection."""
        print("[VAD] Speech started")
        self._speaking = True
        self._silence_start = None
        
        # Send interrupt frame to notify downstream components
        interrupt_frame = InterruptFrame(
            frame_type_string="vad_interrupt",
            reason="speech_detected"
        )
        self._output.send(interrupt_frame)
        
        # Start current segment with pre-speech buffer
        self._current_segment = bytearray(self._pre_buffer)
        self._pre_buffer = bytearray()

    def _finalize_segment(self) -> None:
        """Finalize the current speech segment and output it."""
        if not self._current_segment:
            return
            
        segment_bytes = bytes(self._current_segment)
        self._current_segment = bytearray()
        self._speaking = False
        self._silence_start = None
        self._vad_iterator.reset_states()
        
        if len(segment_bytes) >= self._min_speech_bytes:
            # Create output frame with the complete speech segment
            output_frame = AudioFrame(
                frame_type_string="vad_speech_segment",
                pcm16_data=segment_bytes,
                sample_rate=self._sample_rate,
                channels=1
            )
            self._output.send(output_frame)
            print(f"[VAD] Speech segment finalized: {len(segment_bytes)} bytes")
        else:
            print(f"[VAD] Segment too short: {len(segment_bytes)} bytes")

    def run(self) -> None:
        """Main processing loop."""
        print("[VAD] Starting Voice Activity Detection")
        
        for frame in self._input_channel.stream(self.stop_event):
            if frame is None:
                break
            
            if isinstance(frame, AudioFrame):
                self._process_audio_frame(frame)
        
        # Clean up any remaining segment on stop
        if self._current_segment:
            self._finalize_segment()
        
        print("[VAD] Voice Activity Detection stopped")
