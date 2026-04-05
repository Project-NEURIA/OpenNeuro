"""
TTFA (Time to First Audio) Profiler for the OpenNeuro voice pipeline.

Two modes:
  --mode sequential   Run each component one-at-a-time, cProfile each (default)
  --mode pipeline     Run full pipeline in parallel, wall-clock timestamps only

Usage:
    cd backend
    uv run --no-group cuda python -m profiling.ttfa_profile --mode sequential
    uv run --no-group cuda python -m profiling.ttfa_profile --mode pipeline
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import threading
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
from dotenv import load_dotenv  # type: ignore[import-untyped]
from requests import Response

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.channel import Channel, Receiver, Sender  # noqa: E402
from src.core.component import ThreadedComponent, Tag  # noqa: E402
from src.core.frames import AudioFrame, TextFrame, EOS, MessageFrame  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
BACKEND = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

_profilers: dict[str, list[cProfile.Profile]] = defaultdict(list)
_timestamps: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


def _import_mod(name: str, filepath: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_modules():
    vad_mod = _import_mod("src.lib.audio.vad", str(BACKEND / "src/lib/audio/vad.py"))
    asr_mod = _import_mod("src.lib.audio.asr", str(BACKEND / "src/lib/audio/asr.py"))
    llm_mod = _import_mod("src.lib.llm.llm", str(BACKEND / "src/lib/llm/llm.py"))
    tts_mod = _import_mod("src.lib.audio.tts", str(BACKEND / "src/lib/audio/tts.py"))
    return vad_mod, asr_mod, llm_mod, tts_mod


def _load_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """Load WAV file, return (data as (1, samples) float32, sample_rate)."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return data.reshape(1, -1), sr


def _print_cprofile(
    name: str, profiler_list: list[cProfile.Profile], report: list[str]
):
    if not profiler_list:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg: str = ""):
        print(msg)
        report.append(msg)

    sep = "=" * 80
    log(sep)
    log(f"cProfile: {name}")
    log(sep)

    stream = io.StringIO()
    merged = None
    for p in profiler_list:
        try:
            if merged is None:
                merged = pstats.Stats(p, stream=stream)
            else:
                merged.add(p)
        except TypeError:
            continue

    if merged is None:
        log("  (no data)")
        log()
        return

    prof_path = RESULTS_DIR / f"{name.replace('.', '_')}.prof"
    for p in profiler_list:
        try:
            p.dump_stats(str(prof_path))
            break
        except Exception:
            continue
    log(f"  Saved: {prof_path}")

    merged.sort_stats("cumulative")
    merged.print_stats(20)
    log(stream.getvalue())


# ──────────────────────────────────────────────────────────────────────
# SEQUENTIAL MODE — run each component alone, cProfile each
# ──────────────────────────────────────────────────────────────────────


def run_sequential(wav_path: str):
    print("=" * 60)
    print("MODE: SEQUENTIAL (cProfile each component individually)")
    print("=" * 60)
    print()

    vad_mod, asr_mod, llm_mod, tts_mod = _load_modules()
    VAD, VADConfig = vad_mod.VAD, vad_mod.VADConfig
    ASR, ASRConfig = asr_mod.ASR, asr_mod.ASRConfig
    TTSConfig = tts_mod.TTSConfig

    audio_data, sr = _load_wav(wav_path)
    chunk_samples = int(sr * 0.02)
    report: list[str] = []

    def log(msg: str = ""):
        print(msg)
        report.append(msg)

    # ── 1. VAD._process_audio_frame ──
    log("── 1. Profiling VAD._process_audio_frame ──")
    vad = VAD(config=VADConfig())
    vad._stop_event = threading.Event()
    vad_out_channel: Channel[AudioFrame] = Channel()
    vad_out_sender = Sender(vad_out_channel)
    vad_outputs = vad_mod.VADOutputs(audio=vad_out_sender)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    for start in range(0, audio_data.shape[1], chunk_samples):
        chunk = audio_data[:, start : start + chunk_samples]
        frame = AudioFrame.new(data=chunk, sample_rate=sr, channels=1)
        vad._process_audio_frame(frame, vad_outputs)
    # Finalize any pending segment
    if vad._current_segment:
        vad._finalize_segment(vad_outputs)
    profiler.disable()
    t1 = time.perf_counter()
    _profilers["VAD._process_audio_frame"].append(profiler)

    # Collect VAD output frames
    vad_frames: list[AudioFrame] = list(vad_out_channel._items)

    n_chunks = audio_data.shape[1] // chunk_samples
    log(
        f"  {n_chunks} chunks processed in {(t1 - t0) * 1000:.0f}ms ({(t1 - t0) * 1000 / n_chunks:.1f}ms/chunk)"
    )
    log(f"  VAD output: {len(vad_frames)} audio segment(s)")
    log()

    if not vad_frames:
        log("  ERROR: VAD produced no segments. Cannot continue.")
        return

    # ── 2. ASR._transcribe_audio ──
    log("── 2. Profiling ASR._transcribe_audio ──")
    asr = ASR(config=ASRConfig())
    asr_results: list[TextFrame] = []

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    for vad_frame in vad_frames:
        result = asr._transcribe_audio(vad_frame)
        if result:
            asr_results.append(result)
    profiler.disable()
    t1 = time.perf_counter()
    _profilers["ASR._transcribe_audio"].append(profiler)

    log(
        f"  {len(vad_frames)} segment(s) transcribed in {(t1 - t0) * 1000:.0f}ms ({(t1 - t0) * 1000 / len(vad_frames):.0f}ms/segment)"
    )
    for r in asr_results:
        log(f"  Transcription: '{r.text}'")
    log()

    if not asr_results:
        log("  ERROR: ASR produced no transcriptions. Cannot continue.")
        return

    # ── 3. LLM TTFT (first streaming token) ──
    log("── 3. Profiling LLM TTFT (litellm.completion → first token) ──")
    from litellm import completion

    # Warmup: pre-load tokenizer so cProfile captures steady-state TTFT
    log("  Warming up litellm tokenizer...")
    try:
        completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            stream=False,
        )
    except Exception:
        pass

    text = asr_results[0].text
    messages = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Keep responses brief.",
        },
        {"role": "user", "content": text},
    ]

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    stream = completion(
        model="groq/llama-3.3-70b-versatile", messages=messages, stream=True
    )
    first_token_text = None
    full_text = ""
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice and choice.delta and choice.delta.content:
            if first_token_text is None:
                profiler.disable()
                t_ttft = time.perf_counter()
                first_token_text = choice.delta.content
            full_text += choice.delta.content
    t1 = time.perf_counter()
    _profilers["LLM.TTFT"].append(profiler)

    ttft_ms = (t_ttft - t0) * 1000 if first_token_text else float("nan")
    log(f"  TTFT: {ttft_ms:.0f}ms")
    log(f"  Total generation: {(t1 - t0) * 1000:.0f}ms, {len(full_text)} chars")
    log(f"  Response: '{full_text[:80]}...'")
    log()

    # ── 4. TTS._worker (requests.post to Inworld) ──
    log("── 4. Profiling TTS requests.post (Inworld API) ──")
    import os
    import json
    import base64
    import requests

    cred = os.getenv("INWORLD_API_KEY")
    tts_config = TTSConfig()
    # Build the sentence to send (simulate StreamFilter output)
    tts_text = full_text.split(".")[0] + "." if "." in full_text else full_text

    # Use a persistent session (matching the improvement we made to TTS)
    session = requests.Session()

    # Warmup: establish TLS connection so cProfile captures steady-state
    log("  Warming up TTS session (TLS handshake)...")
    try:
        warmup_payload = {
            "text": "hi",
            "voice_id": tts_config.voice_id,
            "model_id": tts_config.model_id,
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000},
        }
        warmup_r = session.post(
            tts_config.url,
            json=warmup_payload,
            headers={
                "Authorization": f"Basic {cred}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        warmup_r.close()
    except Exception:
        pass

    headers = {
        "Authorization": f"Basic {cred}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": tts_text,
        "voice_id": tts_config.voice_id,
        "model_id": tts_config.model_id,
        "audio_config": {
            "audio_encoding": "LINEAR16",
            "sample_rate_hertz": 48000,
        },
    }

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    response: Response = session.post(
        tts_config.url,
        json=payload,
        headers=headers,
        stream=True,
        timeout=10,
    )
    response.raise_for_status()
    tts_frames: list[AudioFrame] = []
    t_first_audio = None
    for line in response.iter_lines():
        if not line:
            continue
        msg = json.loads(line)
        raw = base64.b64decode(msg["result"]["audioContent"])
        if len(raw) > 44:
            if t_first_audio is None:
                t_first_audio = time.perf_counter()
            tts_frames.append(
                AudioFrame.new(data=raw[44:], sample_rate=48000, channels=1)
            )
    profiler.disable()
    t1 = time.perf_counter()
    _profilers["TTS.requests.post"].append(profiler)

    ttfb_ms = (t_first_audio - t0) * 1000 if t_first_audio else float("nan")
    log(f"  TTS TTFB: {ttfb_ms:.0f}ms")
    log(f"  Total: {(t1 - t0) * 1000:.0f}ms, {len(tts_frames)} audio chunks")
    log(f"  Input text: '{tts_text}'")
    log()

    # ── Summary ──
    sep = "=" * 80
    log(sep)
    log("SEQUENTIAL PROFILING SUMMARY")
    log(sep)
    log()
    log(f"  {'Function':<30} {'Wall Time':>12}")
    log(f"  {'-' * 30} {'-' * 12}")
    log(f"  {'VAD._process_audio_frame':<30} {(t1 - t0) * 1000:>10.0f}ms")

    # Recompute per-component times from profiler
    for name in [
        "VAD._process_audio_frame",
        "ASR._transcribe_audio",
        "LLM.TTFT",
        "TTS.requests.post",
    ]:
        profs = _profilers.get(name, [])
        if profs:
            s = io.StringIO()
            st = pstats.Stats(profs[0], stream=s)
            total = cast(float, getattr(st, "total_tt", 0.0))
            log(f"  {name:<30} {total * 1000:>10.0f}ms (cProfile)")

    log()
    log("  Estimated TTFA = ASR + LLM TTFT + TTS TTFB")
    asr_time = (
        sum(p.getstats()[0].totaltime for p in _profilers["ASR._transcribe_audio"])
        if _profilers["ASR._transcribe_audio"]
        else 0
    )
    log(
        f"                ≈ {asr_time * 1000:.0f} + {ttft_ms:.0f} + {ttfb_ms:.0f} = {asr_time * 1000 + ttft_ms + ttfb_ms:.0f}ms"
    )
    log()

    # ── cProfile details ──
    for name in [
        "VAD._process_audio_frame",
        "ASR._transcribe_audio",
        "LLM.TTFT",
        "TTS.requests.post",
    ]:
        _print_cprofile(name, _profilers.get(name, []), report)

    report_path = RESULTS_DIR / "report_sequential.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport saved to: {report_path}")


# ──────────────────────────────────────────────────────────────────────
# PIPELINE MODE — full parallel pipeline, wall-clock timestamps
# ──────────────────────────────────────────────────────────────────────


class FileSourceOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class FileSource(ThreadedComponent[tuple[()], FileSourceOutputs]):
    tags = Tag(io={"source"}, functionality={"audio"})
    _registerable = False

    def __init__(self, wav_path: str) -> None:
        super().__init__()
        self._wav_path = wav_path

    def run(self, inputs: tuple[()], outputs: FileSourceOutputs) -> None:
        data, sr = _load_wav(self._wav_path)
        chunk_samples = int(sr * 0.02)
        print(f"[FileSource] {data.shape[1]} samples at {sr}Hz")

        with _lock:
            _timestamps["pipeline:audio_start"].append(time.perf_counter())

        for start in range(0, data.shape[1], chunk_samples):
            if self.stop_event.is_set():
                break
            chunk = data[:, start : start + chunk_samples]
            outputs.audio.send(AudioFrame.new(data=chunk, sample_rate=sr, channels=1))
            time.sleep(chunk_samples / sr)

        print("[FileSource] Done")
        time.sleep(3.0)
        self.stop_event.set()


class NullSinkInputs(NamedTuple):
    audio: Receiver[AudioFrame]


class NullSink(ThreadedComponent[NullSinkInputs, tuple[()]]):
    tags = Tag(io={"sink"}, functionality={"audio"})
    _registerable = False

    def __init__(self) -> None:
        super().__init__()
        self._count = 0

    def run(self, inputs: NullSinkInputs, outputs: tuple[()]) -> None:
        for frame in inputs.audio:
            if frame is None:
                break
            self._count += 1
            if self._count == 1:
                with _lock:
                    _timestamps["pipeline:first_audio"].append(time.perf_counter())
                print("[NullSink] First audio received!")
        print(f"[NullSink] {self._count} frames total")


class _Stub(ThreadedComponent[Any, Any]):
    tags = Tag(io={"conduit"}, functionality={"misc"})
    _registerable = False

    def run(self, inputs, outputs):
        pass


def run_pipeline(wav_path: str):
    print("=" * 60)
    print("MODE: PIPELINE (full parallel, wall-clock timestamps)")
    print("=" * 60)
    print()

    vad_mod, asr_mod, llm_mod, tts_mod = _load_modules()
    VAD, VADConfig = vad_mod.VAD, vad_mod.VADConfig
    ASR, ASRConfig = asr_mod.ASR, asr_mod.ASRConfig
    LLM, LLMConfig = llm_mod.LLM, llm_mod.LLMConfig
    TTS, TTSConfig = tts_mod.TTS, tts_mod.TTSConfig

    # Timestamp patches (no cProfile)
    orig_vad = VAD._process_audio_frame

    def ts_vad(self, frame, outputs):
        t = time.perf_counter()
        with _lock:
            _timestamps["VAD._process_audio_frame:enter"].append(t)
        result = orig_vad(self, frame, outputs)
        with _lock:
            _timestamps["VAD._process_audio_frame:exit"].append(time.perf_counter())
        return result

    VAD._process_audio_frame = ts_vad

    orig_finalize = VAD._finalize_segment

    def ts_finalize(self, outputs):
        result = orig_finalize(self, outputs)
        with _lock:
            _timestamps["VAD._finalize_segment:exit"].append(time.perf_counter())
        return result

    VAD._finalize_segment = ts_finalize

    orig_asr = ASR._transcribe_audio

    def ts_asr(self, frame):
        t = time.perf_counter()
        with _lock:
            _timestamps["ASR._transcribe_audio:enter"].append(t)
        result = orig_asr(self, frame)
        with _lock:
            _timestamps["ASR._transcribe_audio:exit"].append(time.perf_counter())
        return result

    ASR._transcribe_audio = ts_asr

    orig_llm_run = LLM.run

    def ts_llm(self, inputs, outputs):
        with _lock:
            _timestamps["LLM.run:enter"].append(time.perf_counter())
        orig_send = outputs.token.send
        first = threading.Event()

        def wrap_send(item):
            if (
                not first.is_set()
                and isinstance(item, TextFrame)
                and not isinstance(item, EOS)
            ):
                first.set()
                with _lock:
                    _timestamps["LLM.run:first_token"].append(time.perf_counter())
            return orig_send(item)

        outputs.token.send = wrap_send
        result = orig_llm_run(self, inputs, outputs)
        with _lock:
            _timestamps["LLM.run:exit"].append(time.perf_counter())
        return result

    LLM.run = ts_llm

    orig_tts = TTS._worker

    def ts_tts(self, outputs):
        with _lock:
            _timestamps["TTS._worker:enter"].append(time.perf_counter())
        orig_audio = outputs.audio.send
        first = threading.Event()

        def wrap_audio(item):
            if not first.is_set():
                first.set()
                with _lock:
                    _timestamps["TTS._worker:first_audio"].append(time.perf_counter())
            return orig_audio(item)

        outputs.audio.send = wrap_audio
        result = orig_tts(self, outputs)
        with _lock:
            _timestamps["TTS._worker:exit"].append(time.perf_counter())
        return result

    TTS._worker = ts_tts

    # ch1: FileSource → VAD (audio)
    # ch2: VAD → ASR (audio)
    # ch3: ASR → Adapter (text)
    # ch4: Adapter → LLM (messages)
    # ch5: LLM → TTS (tokens)
    # ch6: TTS → NullSink (audio)
    ch1: Channel[AudioFrame] = Channel()
    ch2: Channel[AudioFrame] = Channel()
    ch3: Channel[TextFrame] = Channel()
    ch4: Channel[list[MessageFrame]] = Channel()
    ch5: Channel[TextFrame] = Channel()
    ch6: Channel[AudioFrame] = Channel()

    adapter_sender_2 = Sender(ch4)

    def adapter3():
        comp = _Stub()
        comp._stop_event = threading.Event()
        for tf in Receiver(ch3, threading.Event()):
            if tf is None:
                break
            msgs = [
                MessageFrame.new(
                    role="system",
                    content="You are a helpful voice assistant. Keep responses brief.",
                ),
                MessageFrame.new(role="user", content=tf.text),
            ]
            adapter_sender_2.send(msgs)
            print(f"[Adapter] '{tf.text}'")

    threading.Thread(target=adapter3, daemon=True).start()

    null_sink_2 = NullSink()
    tts_2 = TTS(config=TTSConfig())
    llm_2 = LLM(config=LLMConfig())
    asr_2 = ASR(config=ASRConfig())
    vad_2 = VAD(config=VADConfig())
    file_source_2 = FileSource(wav_path=wav_path)

    null_sink_2.start(NullSinkInputs(audio=Receiver(ch6, threading.Event())), ())
    tts_2.start(
        tts_mod.TTSInputs(text=Receiver(ch5, threading.Event())),
        tts_mod.TTSOutputs(audio=Sender(ch6), text=Sender()),
    )
    llm_2.start(
        llm_mod.LLMInputs(messages=Receiver(ch4, threading.Event())),
        llm_mod.LLMOutputs(token=Sender(ch5)),
    )
    asr_2.start(
        asr_mod.ASRInputs(audio=Receiver(ch2, threading.Event())),
        asr_mod.ASROutputs(text=Sender(ch3)),
    )
    vad_2.start(
        vad_mod.VADInputs(audio=Receiver(ch1, threading.Event())),
        vad_mod.VADOutputs(audio=Sender(ch2)),
    )
    file_source_2.start((), FileSourceOutputs(audio=Sender(ch1)))

    print("[Pipeline] Running...")
    deadline = time.time() + 60
    while time.time() < deadline:
        if file_source_2.stop_event.is_set():
            time.sleep(5.0)
            break
        time.sleep(0.5)

    file_source_2.stop()
    vad_2.stop()
    asr_2.stop()
    llm_2.stop()
    tts_2.stop()
    null_sink_2.stop()
    time.sleep(1.0)

    # Report
    report: list[str] = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    sep = "=" * 80
    log(sep)
    log("PIPELINE TTFA REPORT")
    log(sep)
    log()

    vad_exit = _timestamps.get("VAD._finalize_segment:exit", [None])[0]
    asr_exit = _timestamps.get("ASR._transcribe_audio:exit", [None])[0]
    llm_ft = _timestamps.get("LLM.run:first_token", [None])[0]
    tts_fa = _timestamps.get("TTS._worker:first_audio", [None])[0]

    def ms(a, b):
        if a and b:
            return f"{(b - a) * 1000:.0f}ms"
        return "N/A"

    log(f"  ASR latency (VAD-end → ASR done):    {ms(vad_exit, asr_exit)}")
    log(f"  LLM TTFT (ASR done → first token):   {ms(asr_exit, llm_ft)}")
    log(f"  TTS TTFB (first token → first audio): {ms(llm_ft, tts_fa)}")
    log(f"  TTFA (VAD-end → first audio):         {ms(vad_exit, tts_fa)}")
    log()

    # Per-component wall times
    log(f"  {'Component':<30} {'Calls':>6} {'Avg (ms)':>10}")
    log(f"  {'-' * 30} {'-' * 6} {'-' * 10}")
    for name in ["VAD._process_audio_frame", "ASR._transcribe_audio"]:
        enters = _timestamps.get(f"{name}:enter", [])
        exits = _timestamps.get(f"{name}:exit", [])
        n = min(len(enters), len(exits))
        if n:
            avg = sum((exits[i] - enters[i]) * 1000 for i in range(n)) / n
            log(f"  {name:<30} {n:>6} {avg:>8.1f}ms")
    log()

    report_path = RESULTS_DIR / "report_pipeline.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport saved to: {report_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TTFA Profiler")
    parser.add_argument(
        "--mode", choices=["sequential", "pipeline"], default="sequential"
    )
    args = parser.parse_args()

    wav_path = str(Path(__file__).parent / "assets" / "test_audio.wav")
    if not Path(wav_path).exists():
        from .generate_test_audio import generate  # type: ignore[import-not-found]

        generate()

    if args.mode == "sequential":
        run_sequential(wav_path)
    else:
        run_pipeline(wav_path)


if __name__ == "__main__":
    main()
