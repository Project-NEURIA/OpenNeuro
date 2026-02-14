from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperFeatureExtractor

from openneuro.core.component import Component
from openneuro.core.channel import Channel


class VAD(Component[Channel[bytes]]):
    def __init__(
        self,
        *,
        silence_seconds: float = 0.9,
        max_silence_seconds: float = 1.4,
        pre_speech_seconds: float = 1.0,
        min_speech_seconds: float = 0.5,
        turn_threshold: float = 0.89,
        smart_turn_onnx: str = str(Path(__file__).resolve().parents[3] / "assets" / "smart-turn-v3.0.onnx"),
    ) -> None:
        super().__init__()
        self._output = Channel[bytes]()

        self._silence_seconds = silence_seconds
        self._max_silence_seconds = max_silence_seconds
        self._turn_threshold = turn_threshold
        self._min_speech_bytes = int(48000 * 2 * min_speech_seconds)
        self._max_pre_bytes = int(48000 * 2 * pre_speech_seconds)

        self._speaking = False
        self._silence_start: float | None = None
        self._pre = bytearray()
        self._cur = bytearray()
        self._vad_buf: list[float] = []

        self._load_silero()
        self._load_smart_turn(smart_turn_onnx)

    def get_output_channels(self) -> tuple[Channel[bytes]]:
        return (self._output,)

    def set_input_channels(self, t1: Channel[bytes]) -> None:
        self._input_channel = t1

    def _load_silero(self) -> None:
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (_, _, _, vad_iterator_cls, _) = utils
        self._it = vad_iterator_cls(self._model)

    def _load_smart_turn(self, onnx_path: str) -> None:
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._smart_turn_session = ort.InferenceSession(onnx_path, sess_options=so)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    def _check_smart_turn(self) -> bool:
        seg = bytes(self._cur)
        if not seg:
            return False

        pcm = np.frombuffer(seg, dtype=np.int16)
        pcm16k = pcm[::3].astype(np.float32) / 32768.0

        max_samples = 8 * 16000
        if len(pcm16k) > max_samples:
            pcm16k = pcm16k[-max_samples:]
        else:
            pcm16k = np.pad(pcm16k, (max_samples - len(pcm16k), 0), mode="constant")

        inputs = self._feature_extractor(
            pcm16k,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=max_samples,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)
        outputs = self._smart_turn_session.run(None, {"input_features": input_features})
        return outputs[0][0].item() > self._turn_threshold

    def run(self) -> None:
        for data in self._input_channel.stream(self.stop_event):
            if data is None:
                break
            pcm = np.frombuffer(data, dtype=np.int16)
            mono16k = pcm[::3].astype(np.float32) / 32768.0

            self._vad_buf.extend(mono16k.tolist())

            while len(self._vad_buf) >= 512:
                chunk = torch.tensor(self._vad_buf[:512])
                self._vad_buf = self._vad_buf[512:]
                out = self._it(chunk, return_seconds=False)

                start = out and "start" in out
                end = out and "end" in out

                if start and not self._speaking:
                    self._speaking = True
                    self._silence_start = None
                    self._cur = bytearray(self._pre)
                    self._pre = bytearray()

                if self._speaking and end:
                    if self._silence_start is None:
                        self._silence_start = time.time()

            if self._speaking:
                self._cur.extend(data)

                if self._silence_start is not None:
                    silence_duration = time.time() - self._silence_start
                    if silence_duration >= self._max_silence_seconds:
                        self._finalize()
                    elif silence_duration >= self._silence_seconds:
                        if self._check_smart_turn():
                            self._finalize()
            else:
                self._pre.extend(data)
                if len(self._pre) > self._max_pre_bytes:
                    self._pre = self._pre[-self._max_pre_bytes :]

    def _finalize(self) -> None:
        seg = bytes(self._cur)
        self._cur = bytearray()
        self._speaking = False
        self._silence_start = None
        self._it.reset_states()

        if len(seg) >= self._min_speech_bytes:
            self._output.send(seg)
