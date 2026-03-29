"""Trace where time is spent between VAD-end and LLM first token.

Measures each hop in the chain:
  VAD._finalize_segment sends AudioFrame
  → Channel transit to ASR
  → ASR._transcribe_audio (Groq Whisper API)
  → Channel transit to Adapter
  → Adapter builds message list
  → Channel transit to LLM
  → litellm.completion() TTFT
"""

import time
import sys
import threading
from pathlib import Path
from collections import defaultdict
from typing import Any, NamedTuple

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.channel import Channel, Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioFrame, TextFrame, EOS, MessageFrame

_lock = threading.Lock()
_ts: dict[str, float] = {}


def ts(name: str):
    """Record timestamp — only keeps the FIRST occurrence of each name."""
    with _lock:
        if name not in _ts:
            _ts[name] = time.perf_counter()


def _import_mod(name, filename):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


backend = Path(__file__).resolve().parent.parent
vad_mod = _import_mod("src.core.conduit.vad", str(backend / "src/core/conduit/vad.py"))
asr_mod = _import_mod("src.core.conduit.asr", str(backend / "src/core/conduit/asr.py"))
llm_mod = _import_mod("src.core.conduit.llm", str(backend / "src/core/conduit/llm.py"))

VAD, VADConfig = vad_mod.VAD, vad_mod.VADConfig
ASR, ASRConfig = asr_mod.ASR, asr_mod.ASRConfig
LLM, LLMConfig = llm_mod.LLM, llm_mod.LLMConfig
VADInputs, VADOutputs = vad_mod.VADInputs, vad_mod.VADOutputs
ASRInputs, ASROutputs = asr_mod.ASRInputs, asr_mod.ASROutputs
LLMInputs, LLMOutputs = llm_mod.LLMInputs, llm_mod.LLMOutputs


# -- Monkey-patch to add timestamps --

# VAD._finalize_segment: record when audio is sent
orig_finalize = VAD._finalize_segment


def patched_finalize(self, outputs):
    ts("vad_finalize_enter")
    result = orig_finalize(self, outputs)
    ts("vad_finalize_exit")
    return result


VAD._finalize_segment = patched_finalize

# ASR._transcribe_audio: record API call time
orig_transcribe = ASR._transcribe_audio


def patched_transcribe(self, frame):
    ts("asr_transcribe_enter")
    result = orig_transcribe(self, frame)
    ts("asr_transcribe_exit")
    return result


ASR._transcribe_audio = patched_transcribe

# ASR._worker_loop: record when frame is dequeued
orig_worker = ASR._worker_loop


def patched_worker(self, outputs):
    orig_get = self._task_queue.get
    first_get = threading.Event()

    def timed_get(*a, **kw):
        item = orig_get(*a, **kw)
        if item is not None and not first_get.is_set():
            first_get.set()
            ts("asr_queue_get")
        return item

    self._task_queue.get = timed_get
    return orig_worker(self, outputs)


ASR._worker_loop = patched_worker

# LLM: record when messages arrive and when completion is called
orig_llm_run = LLM.run


def patched_llm_run(self, inputs, outputs):
    orig_send = outputs.token.send
    first_token = threading.Event()

    def token_send(item):
        if (
            not first_token.is_set()
            and isinstance(item, TextFrame)
            and not isinstance(item, EOS)
        ):
            first_token.set()
            ts("llm_first_token")
        return orig_send(item)

    outputs.token.send = token_send

    # Patch the messages receiver to record when first message arrives
    orig_iter = inputs.messages.__call__

    def patched_iter(subscriber, **kw):
        it = orig_iter(subscriber, **kw)
        first_msg = threading.Event()
        orig_next = it.__next__

        def timed_next():
            item = orig_next()
            if item is not None and not first_msg.is_set():
                first_msg.set()
                ts("llm_msg_received")
            return item

        it.__next__ = timed_next
        return it

    inputs.messages.__call__ = patched_iter

    return orig_llm_run(self, inputs, outputs)


LLM.run = patched_llm_run


# -- FileSource: send pre-recorded audio --

import wave
import numpy as np


class FileSourceOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class FileSource(ThreadedComponent[tuple[()], FileSourceOutputs]):
    tags = Tag(io={"source"}, functionality={"audio"})

    def __init__(self, wav_path: str):
        super().__init__()
        self._wav_path = wav_path

    def run(self, inputs, outputs):
        with wave.open(self._wav_path, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        data = data.reshape(1, -1)
        chunk_samples = int(sr * 0.02)
        for start in range(0, data.shape[1], chunk_samples):
            if self.stop_event.is_set():
                break
            chunk = data[:, start : start + chunk_samples]
            outputs.audio.send(AudioFrame.new(data=chunk, sample_rate=sr, channels=1))
            time.sleep(chunk_samples / sr)
        time.sleep(3.0)
        self.stop_event.set()


class _Adapter(ThreadedComponent[Any, Any]):
    tags = Tag(io={"conduit"}, functionality={"misc"})
    _registerable = False

    def run(self, inputs, outputs):
        pass


def run_test(model: str):
    _ts.clear()
    print(f"\n{'=' * 60}")
    print(f"Model: {model}")
    print(f"{'=' * 60}")

    wav_path = str(Path(__file__).parent / "assets" / "test_audio.wav")

    file_source = FileSource(wav_path=wav_path)
    vad = VAD(config=VADConfig())
    asr = ASR(config=ASRConfig())
    llm = LLM(config=LLMConfig(model=model))

    # Wire: FileSource -> VAD -> ASR -> Adapter -> LLM -> NullSink
    ch1 = Channel()
    s1 = Sender(ch1)
    r1 = Receiver(ch1)
    ch2 = Channel()
    s2 = Sender(ch2)
    r2 = Receiver(ch2)
    ch3 = Channel()
    s3 = Sender(ch3)
    r3 = Receiver(ch3)
    ch4 = Channel()
    s4 = Sender(ch4)
    r4 = Receiver(ch4)

    # Adapter thread
    def adapter():
        comp = _Adapter()
        comp._stop_event = threading.Event()
        for text_frame in Receiver(ch3)(comp):
            if text_frame is None:
                break
            if comp.stop_event.is_set():
                break
            ts("adapter_received")
            msgs = [
                MessageFrame.new(
                    role="system",
                    content="You are a helpful voice assistant. Keep responses brief.",
                ),
                MessageFrame.new(role="user", content=text_frame.text),
            ]
            s4.send(msgs)
            ts("adapter_sent")

    threading.Thread(target=adapter, daemon=True).start()

    # LLM input from adapter (ch4), output tokens to ch5
    ch5 = Channel()
    s5 = Sender(ch5)

    # Start components
    llm.start(LLMInputs(messages=r4), LLMOutputs(token=s5))
    asr.start(ASRInputs(audio=r2), ASROutputs(text=Sender(ch3)))
    vad.start(VADInputs(audio=r1), VADOutputs(audio=s2))
    file_source.start((), FileSourceOutputs(audio=s1))

    # Wait
    deadline = time.time() + 30
    while time.time() < deadline:
        if "llm_first_token" in _ts:
            break
        time.sleep(0.1)

    time.sleep(1)
    file_source.stop()
    vad.stop()
    asr.stop()
    llm.stop()
    time.sleep(0.5)

    # Print results
    def delta(a, b):
        if a in _ts and b in _ts:
            return f"{(_ts[b] - _ts[a]) * 1000:.0f}ms"
        return "N/A"

    print(
        f"  VAD finalize:                    {delta('vad_finalize_enter', 'vad_finalize_exit')}"
    )
    print(
        f"  VAD exit → ASR queue get:        {delta('vad_finalize_exit', 'asr_queue_get')}"
    )
    print(
        f"  ASR queue get → transcribe enter: {delta('asr_queue_get', 'asr_transcribe_enter')}"
    )
    print(
        f"  ASR transcribe (API call):       {delta('asr_transcribe_enter', 'asr_transcribe_exit')}"
    )
    print(
        f"  ASR exit → Adapter received:     {delta('asr_transcribe_exit', 'adapter_received')}"
    )
    print(
        f"  Adapter received → sent:         {delta('adapter_received', 'adapter_sent')}"
    )
    print(
        f"  Adapter sent → LLM msg received: {delta('adapter_sent', 'llm_msg_received')}"
    )
    print(
        f"  LLM msg received → first token:  {delta('llm_msg_received', 'llm_first_token')}"
    )
    print(f"  ─────────────────────────────────")
    print(
        f"  VAD exit → LLM first token:      {delta('vad_finalize_exit', 'llm_first_token')}"
    )


if __name__ == "__main__":
    run_test("groq/llama-3.3-70b-versatile")
    run_test("openai/gpt-5.4")
