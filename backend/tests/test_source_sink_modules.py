from __future__ import annotations

import importlib
import sys
import types

import numpy as np

from src.core.channel import Sender, Channel
from src.core.frames import TextFrame, VideoFrame, VideoDataFormat, AudioFrame
from src.core.sink.do_nothing import DoNothing, DoNothingInputs
from src.core.sink.text_display import TextDisplay, TextDisplayInputs, TextDisplayOutputs
from src.core.sink.video_stream import VideoStream, VideoStreamInputs, VideoStreamOutputs
from src.core.sink.speaker import Speaker, SpeakerConfig, SpeakerInputs
from src.core.source.prompt_repeater import (
    PromptRepeater,
    PromptRepeaterConfig,
    PromptRepeaterOutputs,
)
from src.core.source.pulse import Pulse, PulseConfig, PulseOutputs
from src.core.source.dummy_poses import DummyPosesInput, DummyPosesConfig, DummyPosesOutputs
from src.core.source.text_input import TextInput, TextInputInputs, TextInputOutputs
from src.core.source.video_source import VideoSource, VideoSourceConfig, VideoSourceOutputs
from src.core.source.video_player import VideoPlayer, VideoPlayerConfig
from src.core.source.camera import Camera
from src.core.source.mic import Mic, MicConfig, MicOutputs
from src.core.source.openvr_player import OpenVRPlayer, OpenVRPlayerConfig
from src.core.source.vrchat import VRChatVideo, VRChatVideoOutputs


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def test_do_nothing_text_display_text_input() -> None:
    d = DoNothing[TextFrame]()
    d.run(DoNothingInputs(input=_FakeRecv([TextFrame.new(text="x"), None])), ())

    sent = []
    td = TextDisplay()
    td.run(
        TextDisplayInputs(text=_FakeRecv([TextFrame.new(text="a"), None])),
        TextDisplayOutputs(ui_text=types.SimpleNamespace(send=lambda x: sent.append(x))),
    )
    assert sent and sent[0].get() == "a"

    sent2 = []
    ti = TextInput()
    ti.run(
        TextInputInputs(ui_text=_FakeRecv([TextFrame.new(text="b"), None])),
        TextInputOutputs(text=types.SimpleNamespace(send=lambda x: sent2.append(x))),
    )
    assert sent2 and sent2[0].get() == "b"


def test_video_stream_and_speaker(monkeypatch) -> None:
    fake_cv2 = types.SimpleNamespace(
        IMWRITE_JPEG_QUALITY=1,
        imencode=lambda ext, arr, params: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    sent = []
    vs = VideoStream()
    vf = VideoFrame.new(data=np.zeros((2, 2, 3), dtype=np.uint8), format=VideoDataFormat.BGR)
    vs.run(
        VideoStreamInputs(video=_FakeRecv([vf, None])),
        VideoStreamOutputs(ui_video=types.SimpleNamespace(send=lambda b: sent.append(b))),
    )
    assert sent[0] == b"\x01\x02\x03"

    writes = []

    class _Out:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, arr):
            writes.append(arr)

    fake_sd = types.SimpleNamespace(OutputStream=lambda **kwargs: _Out())
    monkeypatch.setattr("src.core.sink.speaker.sd", fake_sd)
    sp = Speaker(SpeakerConfig(sample_rate=16000, channels=1))
    af = AudioFrame.new(data=np.zeros((1, 8), dtype=np.float32), sample_rate=16000, channels=1)
    sp.run(SpeakerInputs(audio=_FakeRecv([af, None])), ())
    assert len(writes) == 1


def test_prompt_pulse_dummy_poses() -> None:
    psent = []
    pr = PromptRepeater(PromptRepeaterConfig(prompt="p", interval_seconds=0.0))
    orig_wait = pr.stop_event.wait

    def _w_pr(t):
        pr.stop_event.set()
        return orig_wait(0)

    pr.stop_event.wait = _w_pr  # type: ignore[method-assign]
    pr.run((), PromptRepeaterOutputs(text=types.SimpleNamespace(send=lambda x: psent.append(x))))
    assert psent and psent[0].get() == "p"

    pulsed = []
    pu = Pulse(PulseConfig(interval=0.0))
    orig_wait2 = pu.stop_event.wait

    def _w_pu(t):
        pu.stop_event.set()
        return orig_wait2(0)

    pu.stop_event.wait = _w_pu  # type: ignore[method-assign]
    pu.run((), PulseOutputs(pulse=types.SimpleNamespace(send=lambda x: pulsed.append(x))))
    assert pulsed

    poses = []
    dp = DummyPosesInput(DummyPosesConfig(fps=1.0))

    def _send(frame):
        poses.append(frame)
        dp.stop_event.set()

    dp.run((), DummyPosesOutputs(poses=types.SimpleNamespace(send=_send)))
    assert poses and poses[0].get()["head"] is not None


def test_video_source_camera_player_and_mic(monkeypatch) -> None:
    class _Cap:
        def __init__(self):
            self.read_calls = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.read_calls += 1
            if self.read_calls == 1:
                return True, np.zeros((4, 6, 3), dtype=np.uint8)
            return False, None

        def get(self, _prop):
            return 30.0

        def release(self):
            self.released = True

        def set(self, *_args):
            return None

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda src: _Cap(),
        CAP_PROP_FPS=1,
        CAP_PROP_POS_FRAMES=2,
        INTER_LINEAR=1,
        resize=lambda frame, size, interpolation: np.zeros((size[1], size[0], 3), dtype=np.uint8),
    )
    monkeypatch.setattr("src.core.source.video_source.cv2", fake_cv2)
    monkeypatch.setattr("src.core.source.camera.cv2", fake_cv2)
    monkeypatch.setattr("src.core.source.video_player.cv2", fake_cv2)

    class _VS(VideoSource):
        def __init__(self, config):
            super().__init__(config)

        def _on_eof(self, cap):
            self.stop_event.set()

    sent = []
    src = _VS(VideoSourceConfig(source="0", width_resize=3, height_resize=2, fps_resample=30))
    ticks = iter([0.0, 2.0, 2.0, 2.0, 2.0])
    monkeypatch.setattr("src.core.source.video_source.time.monotonic", lambda: next(ticks, 2.0))
    monkeypatch.setattr("src.core.source.video_source.time.sleep", lambda _s: None)
    src.run((), VideoSourceOutputs(video=types.SimpleNamespace(send=lambda f: sent.append(f))))
    assert sent and sent[0].width == 3 and sent[0].height == 2
    assert src._resize(np.zeros((2, 4, 3), dtype=np.uint8)).shape[1] == 3
    src._config.width_resize = 4
    src._config.height_resize = None
    assert src._resize(np.zeros((2, 2, 3), dtype=np.uint8)).shape[1] == 4
    src._config.width_resize = None
    src._config.height_resize = 4
    assert src._resize(np.zeros((2, 2, 3), dtype=np.uint8)).shape[0] == 4
    src._config.width_resize = None
    src._config.height_resize = None
    assert src._resize(np.zeros((2, 2, 3), dtype=np.uint8)).shape == (2, 2, 3)
    src._config.source = "abc"
    assert src._open_capture() is not None

    cam_opts = Camera.get_options({})
    assert "config" in cam_opts

    class _Cam:
        def __init__(self, i, n):
            self.index = i
            self.name = n

    fake_enum = types.SimpleNamespace(enumerate_cameras=lambda: [_Cam(0, "front")])
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", fake_enum)
    cam_opts2 = Camera.get_options({})
    assert cam_opts2["config"]["source"][0]["value"] == "0"
    monkeypatch.delitem(sys.modules, "cv2_enumerate_cameras", raising=False)
    cam_opts3 = Camera.get_options({})
    assert "config" in cam_opts3

    vp = VideoPlayer(VideoPlayerConfig(source="x", loop=True))
    c = _Cap()
    vp._on_eof(c)

    class _In:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            mic.stop_event.set()
            return np.zeros((n, 1), dtype=np.int16), None

    fake_sd = types.SimpleNamespace(InputStream=lambda **kwargs: _In())
    monkeypatch.setattr("src.core.source.mic.sd", fake_sd)
    mic = Mic(MicConfig(sample_rate=8000, channels=1, frame_ms=10))
    mout = []
    mic.run((), MicOutputs(audio=types.SimpleNamespace(send=lambda f: mout.append(f))))
    assert mout and isinstance(mout[0], AudioFrame)

    class _BadCap(_Cap):
        def isOpened(self):
            return False

    monkeypatch.setattr(
        "src.core.source.video_source.cv2",
        types.SimpleNamespace(
            VideoCapture=lambda src: _BadCap(),
            CAP_PROP_FPS=1,
            CAP_PROP_POS_FRAMES=2,
            INTER_LINEAR=1,
            resize=fake_cv2.resize,
        ),
    )
    bad_src = _VS(VideoSourceConfig(source="1"))
    try:
        bad_src._open_capture()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_openvr_vrchat_and_package_init(monkeypatch) -> None:
    calls = []

    class _Client:
        def __init__(self, host="h", port=1):
            self.host = host
            self.port = port

        def connect(self):
            calls.append("connect")

        def disconnect(self):
            calls.append("disconnect")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def frame_stream(self):
            frame = types.SimpleNamespace(
                left=np.zeros((2, 2, 4), dtype=np.uint8).tobytes(),
                right=np.zeros((2, 2, 4), dtype=np.uint8).tobytes(),
                width=2,
                height=2,
            )

            class _Ctx:
                def __enter__(self_):
                    return [frame]

                def __exit__(self_, *a):
                    return False

            return _Ctx()

        def get_intrinsics(self):
            return np.eye(3, dtype=np.float32)

        def get_ipd(self):
            return 0.06

        def get_extrinsics(self, frame, eye=0):
            return np.eye(4, dtype=np.float32)

    class _Player:
        def __init__(self, client):
            self.client = client

        def play(self, **kwargs):
            calls.append("play")

    monkeypatch.setattr("src.core.source.openvr_player.Client", _Client)
    monkeypatch.setattr("src.core.source.openvr_player.Player", _Player)
    monkeypatch.setitem(sys.modules, "ovd_client", types.SimpleNamespace(Client=_Client, Player=_Player))
    op = OpenVRPlayer(OpenVRPlayerConfig())
    op.run(types.SimpleNamespace(), types.SimpleNamespace())
    assert calls[:2] == ["connect", "play"]

    out_v = []
    out_c = []
    vc = VRChatVideo(host="h", port=1, fps=1)
    vc.run(
        (),
        VRChatVideoOutputs(
            stereo_video=types.SimpleNamespace(send=lambda x: out_v.append(x)),
            camera_params=types.SimpleNamespace(send=lambda x: out_c.append(x)),
        ),
    )
    assert out_v and out_c

    source_pkg = importlib.import_module("src.core.source")
    sink_pkg = importlib.import_module("src.core.sink")
    assert hasattr(source_pkg, "Camera")
    assert hasattr(sink_pkg, "Speaker")
