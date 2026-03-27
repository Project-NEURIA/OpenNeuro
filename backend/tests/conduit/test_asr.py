from __future__ import annotations

import os
from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.conduit.asr import ASR, ASRConfig, ASRInputs, ASROutputs
from src.core.frames import AudioFrame, TextFrame


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def _audio_frame() -> AudioFrame:
    return AudioFrame.new(
        data=np.zeros((1, 160), dtype=np.float32),
        sample_rate=16000,
        channels=1,
    )


def test_asr_init_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="must be set"):
        ASR(ASRConfig())


def test_asr_prepare_debug_transcribe_worker_and_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    asr = ASR(ASRConfig(timeout=3))

    wav_path = asr._prepare_audio_for_transcription(_audio_frame())
    assert Path(wav_path).exists()
    assert Path(wav_path).read_bytes().startswith(b"RIFF")
    os.unlink(wav_path)

    class _TempFile:
        def __init__(self, path: Path) -> None:
            self.name = str(path)
            path.write_bytes(b"")

        def close(self) -> None:
            return

    bad_path = tmp_path / "broken.wav"
    monkeypatch.setattr(
        "src.core.conduit.asr.tempfile.NamedTemporaryFile",
        lambda suffix, delete: _TempFile(bad_path),
    )
    monkeypatch.setattr(
        "src.core.conduit.asr.wave.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("wave failed")),
    )
    with pytest.raises(RuntimeError, match="wave failed"):
        asr._prepare_audio_for_transcription(_audio_frame())
    assert not bad_path.exists()

    monkeypatch.chdir(tmp_path)
    src_wav = tmp_path / "src.wav"
    src_wav.write_bytes(b"abc")
    asr._save_debug_audio(str(src_wav))
    debug_files = list((tmp_path / "debug").glob("groq_audio_*.wav"))
    assert len(debug_files) == 1
    asr._save_debug_audio(str(tmp_path / "missing.wav"))

    transcribe_path = tmp_path / "transcribe.wav"
    transcribe_path.write_bytes(b"wav")

    class _Response:
        def __init__(
            self, status_code: int, payload: dict[str, object], text: str = "body"
        ):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("bad status")

        def json(self) -> dict[str, object]:
            return self._payload

    monkeypatch.setattr(
        asr, "_prepare_audio_for_transcription", lambda frame: str(transcribe_path)
    )
    monkeypatch.setattr(
        "src.core.conduit.asr.requests.post",
        lambda *args, **kwargs: _Response(200, {"text": " hello world "}),
    )
    result = asr._transcribe_audio(_audio_frame())
    assert result is not None
    assert result.get() == "hello world"
    assert not transcribe_path.exists()

    transcribe_path.write_bytes(b"wav")
    monkeypatch.setattr(
        "src.core.conduit.asr.requests.post",
        lambda *args, **kwargs: _Response(200, {"text": "   "}),
    )
    assert asr._transcribe_audio(_audio_frame()) is None
    assert not transcribe_path.exists()

    monkeypatch.setattr(
        asr,
        "_prepare_audio_for_transcription",
        lambda frame: (_ for _ in ()).throw(RuntimeError("prep failed")),
    )
    assert asr._transcribe_audio(_audio_frame()) is None

    sent = []
    first_frame = _audio_frame()
    second_frame = _audio_frame()
    get_calls = {"count": 0}

    def fake_get(timeout: float):
        idx = get_calls["count"]
        get_calls["count"] += 1
        if idx == 0:
            raise Empty
        if idx == 1:
            return first_frame
        if idx == 2:
            return second_frame
        asr.stop_event.set()
        raise Empty

    transcribe_calls = {"count": 0}

    def fake_transcribe(frame):
        transcribe_calls["count"] += 1
        if transcribe_calls["count"] == 1:
            return TextFrame.new(text="worker text")
        raise RuntimeError("worker boom")

    monkeypatch.setattr(asr._task_queue, "get", fake_get)
    monkeypatch.setattr(asr, "_transcribe_audio", fake_transcribe)
    asr._worker_loop(
        ASROutputs(text=SimpleNamespace(send=lambda value: sent.append(value)))
    )
    assert [frame.get() for frame in sent] == ["worker text"]

    class _FakeThread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            self.joined = False

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.joined = True
            self.timeout = timeout

    monkeypatch.setattr("src.core.conduit.asr.threading.Thread", _FakeThread)
    run_asr = ASR(ASRConfig())
    frame = _audio_frame()
    run_asr.run(
        ASRInputs(audio=_FakeRecv([frame, None])),
        ASROutputs(text=SimpleNamespace(send=lambda value: sent.append(value))),
    )
    assert run_asr._worker_thread is not None
    assert run_asr._worker_thread.started is True
    assert run_asr._worker_thread.joined is True
    queued = run_asr._task_queue.get_nowait()
    assert queued.id == frame.id
