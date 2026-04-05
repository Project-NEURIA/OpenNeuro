from __future__ import annotations

import json
import types

import numpy as np
from fastapi.testclient import TestClient

from src.core.frames import AudioFrame
from src.lib.audio.audio_in import AudioIn, AudioInConfig
from src.lib.audio.audio_out import AudioOut, AudioOutConfig
from src.lib.audio.mic import MicOutputs
from src.lib.audio.remote_audio_in import RemoteAudioIn, RemoteAudioInConfig
from src.lib.audio.remote_audio_out import RemoteAudioOut, RemoteAudioOutConfig
from src.lib.audio.remote_audio_server import (
    RemoteAudioBridgeConfig,
    create_remote_audio_bridge_app,
)
from src.lib.audio.speaker import SpeakerInputs


class _FakeRecv:
    def __init__(self, items: list[object | None]) -> None:
        self._iter = iter(items)

    def __iter__(self) -> "_FakeRecv":
        return self

    def __next__(self) -> object | None:
        return next(self._iter)


def test_audio_in_and_audio_out_options_and_streams(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.lib.audio.audio_in.list_audio_input_devices",
        lambda: [{"value": "7", "label": "CABLE-B Output"}],
    )
    monkeypatch.setattr(
        "src.lib.audio.audio_out.list_audio_output_devices",
        lambda: [{"value": "3", "label": "CABLE-A Input"}],
    )

    assert AudioIn.get_options({}) == {
        "config": {"device": [{"value": "7", "label": "CABLE-B Output"}]}
    }
    assert AudioOut.get_options({}) == {
        "config": {"device": [{"value": "3", "label": "CABLE-A Input"}]}
    }

    class _InputStream:
        def __enter__(self) -> "_InputStream":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, frame_samples: int) -> tuple[np.ndarray, None]:
            audio_in.stop_event.set()
            return np.zeros((frame_samples, 2), dtype=np.int16), None

    monkeypatch.setattr(
        "src.lib.audio.audio_in.sd",
        types.SimpleNamespace(InputStream=lambda **kwargs: _InputStream()),
    )

    captured_frames: list[AudioFrame] = []
    audio_in = AudioIn(
        AudioInConfig(device="7", sample_rate=48000, channels=2, frame_ms=20)
    )
    audio_in.run(
        (),
        MicOutputs(
            audio=types.SimpleNamespace(
                send=lambda frame: captured_frames.append(frame)
            )
        ),
    )
    assert audio_in._device == 7
    assert captured_frames and captured_frames[0].channels == 2

    writes: list[np.ndarray] = []

    class _OutputStream:
        def __enter__(self) -> "_OutputStream":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def write(self, chunk: np.ndarray) -> None:
            writes.append(chunk)

    monkeypatch.setattr(
        "src.lib.audio.audio_out.sd",
        types.SimpleNamespace(OutputStream=lambda **kwargs: _OutputStream()),
    )

    audio_out = AudioOut(AudioOutConfig(device="3", sample_rate=48000, channels=1))
    audio_frame = AudioFrame.new(
        data=np.zeros((1, 960), dtype=np.float32),
        sample_rate=48000,
        channels=1,
    )
    audio_out.run(
        SpeakerInputs(audio=_FakeRecv([audio_frame, None])),
        (),
    )
    assert audio_out._device == 3
    assert len(writes) == 1


def test_remote_audio_in_and_out_handshake(monkeypatch) -> None:
    class _RemoteInWS:
        def __init__(self, component: RemoteAudioIn) -> None:
            self.component = component
            self.sent: list[object] = []
            self._recv_count = 0

        def send(self, payload: object) -> None:
            self.sent.append(payload)

        def recv(self) -> str | bytes:
            self._recv_count += 1
            if self._recv_count == 1:
                return json.dumps(
                    {
                        "ok": True,
                        "role": "audio_in",
                        "sample_rate": 48000,
                        "channels": 2,
                        "frame_ms": 20,
                        "device": "CABLE-B Output",
                    }
                )
            if self._recv_count == 2:
                return b"\x00\x00" * 960 * 2
            self.component.stop_event.set()
            raise RuntimeError("closed")

        def close(self) -> None:
            return None

    remote_in: RemoteAudioIn | None = None
    remote_in_ws: _RemoteInWS | None = None

    def fake_remote_in_connect(*args: object, **kwargs: object) -> _RemoteInWS:
        del args, kwargs
        assert remote_in is not None
        nonlocal remote_in_ws
        remote_in_ws = _RemoteInWS(remote_in)
        return remote_in_ws

    monkeypatch.setattr("src.lib.audio.remote_audio_in.connect", fake_remote_in_connect)

    received_frames: list[AudioFrame] = []
    remote_in = RemoteAudioIn(
        RemoteAudioInConfig(sample_rate=48000, channels=2, frame_ms=20)
    )
    remote_in.run(
        (),
        MicOutputs(
            audio=types.SimpleNamespace(
                send=lambda frame: received_frames.append(frame)
            )
        ),
    )

    assert remote_in_ws is not None
    hello_in = json.loads(remote_in_ws.sent[0])
    assert hello_in["role"] == "audio_in"
    assert received_frames and received_frames[0].channels == 2

    class _RemoteOutWS:
        def __init__(self) -> None:
            self.sent: list[object] = []
            self._recv_count = 0

        def send(self, payload: object) -> None:
            self.sent.append(payload)

        def recv(self) -> str:
            self._recv_count += 1
            assert self._recv_count == 1
            return json.dumps(
                {
                    "ok": True,
                    "role": "audio_out",
                    "sample_rate": 48000,
                    "channels": 1,
                    "frame_ms": 20,
                    "device": "CABLE-A Input",
                }
            )

        def close(self) -> None:
            return None

    remote_out_ws = _RemoteOutWS()
    monkeypatch.setattr(
        "src.lib.audio.remote_audio_out.connect",
        lambda *args, **kwargs: remote_out_ws,
    )

    remote_out = RemoteAudioOut(
        RemoteAudioOutConfig(sample_rate=48000, channels=1, frame_ms=20)
    )
    remote_out.run(
        SpeakerInputs(
            audio=_FakeRecv(
                [
                    AudioFrame.new(
                        data=np.zeros((1, 960), dtype=np.float32),
                        sample_rate=48000,
                        channels=1,
                    ),
                    None,
                ]
            )
        ),
        (),
    )

    hello_out = json.loads(remote_out_ws.sent[0])
    assert hello_out["role"] == "audio_out"
    assert isinstance(remote_out_ws.sent[1], bytes)


def test_remote_audio_bridge_http_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.lib.audio.remote_audio_server.list_audio_input_devices",
        lambda: [{"value": "", "label": "System Default"}],
    )
    monkeypatch.setattr(
        "src.lib.audio.remote_audio_server.list_audio_output_devices",
        lambda: [{"value": "3", "label": "CABLE-A Input"}],
    )

    client = TestClient(
        create_remote_audio_bridge_app(
            RemoteAudioBridgeConfig(
                input_device="CABLE-B Output",
                output_device="CABLE-A Input",
            )
        )
    )

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/devices").json() == {
        "input": [{"value": "", "label": "System Default"}],
        "output": [{"value": "3", "label": "CABLE-A Input"}],
    }
