from __future__ import annotations

import importlib
import os
import sys
import types

from src.core.frames import BodyPoseFrame, BonePose, EOS, InterruptFrame, TextFrame


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def test_openvr_movement_helpers_and_run(monkeypatch) -> None:
    class _Pose:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Client:
        def __init__(self, host="127.0.0.1", port=21213):
            self.host = host
            self.port = port
            self.calls = []

        def connect(self):
            self.calls.append("connect")

        def disconnect(self):
            self.calls.append("disconnect")

        def update_pose(self, **kwargs):
            self.calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules, "ovd_client", types.SimpleNamespace(Client=_Client, Pose=_Pose)
    )
    mod = importlib.reload(importlib.import_module("src.core.sink.openvr_movement"))

    assert mod._to_ovd_pose(None) is None
    converted = mod._to_ovd_pose(
        BonePose(pos_x=1, pos_y=2, pos_z=3, rot_w=1, rot_x=4, rot_y=5, rot_z=6)
    )
    assert converted.pos_z == -3 and converted.rot_x == -4 and converted.rot_z == -6

    client = _Client()
    mod._send_poses(client, {"head": converted})
    assert client.calls[-1]["head"] is converted

    sink = mod.OpenVRMovementSink(mod.OpenVRMovementConfig())
    sink.stop_event.set()
    sink.run(mod.OpenVRMovementInputs(poses=None), mod.OpenVRMovementOutputs())

    sink2 = mod.OpenVRMovementSink(mod.OpenVRMovementConfig())
    frame = BodyPoseFrame(poses={"head": BonePose(), "left_hand": None})
    sink2.run(
        mod.OpenVRMovementInputs(poses=_FakeRecv([frame, None])),
        mod.OpenVRMovementOutputs(),
    )


def test_osc_chatbox_paths(monkeypatch) -> None:
    sent = []

    class _Client:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def send_message(self, address, args):
            sent.append((address, args))

        def close(self):
            sent.append(("close", []))

    monkeypatch.setattr("src.core.sink.osc_chatbox._OscClient", _Client)
    monkeypatch.setattr("src.core.sink.osc_chatbox.time.sleep", lambda _s: None)
    monotonic = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(
        "src.core.sink.osc_chatbox.time.monotonic", lambda: next(monotonic, 3.0)
    )

    from src.core.sink.osc_chatbox import OSCChatbox, OSCChatboxConfig, OSCChatboxInputs

    box = OSCChatbox(OSCChatboxConfig(max_chars=5, text_flush_ms=1, clear_on_last=True))
    assert box._split_text("") == []
    parts = box._split_text("hello world amazing")
    assert parts
    assert box._display_delay("x") >= 1.2
    box._send_chatbox("   ")
    assert sent == []
    box._send_chatbox(" hi ")
    assert sent[-1][1] == ["hi", True, False]
    box._send_chatbox("", reset=True)
    assert sent[-1][1] == ["", True, False]

    box._enqueue_text("hello world")
    assert box._send_queue
    box._flush_text_buffer()

    box._text_buffer = "pending"
    box._flush_text_buffer()
    assert box._send_queue

    box2 = OSCChatbox(OSCChatboxConfig(text_flush_ms=1))
    box2._text_loop(_FakeRecv([TextFrame.new(text="abc"), EOS.END, None]))
    assert box2._send_queue

    box3 = OSCChatbox(OSCChatboxConfig(text_flush_ms=1))
    box3._text_buffer = "abc"
    box3._last_text_time = 0.0
    box3.stop_event.set()
    box3._text_flush_monitor()

    box4 = OSCChatbox(OSCChatboxConfig(clear_on_last=True))
    box4._send_queue.append(("msg", 0.0))
    box4._send_event.set()
    original_send_chatbox = box4._send_chatbox

    def _send_once(text: str, *, reset: bool = False) -> None:
        original_send_chatbox(text, reset=reset)
        box4.stop_event.set()

    box4._send_chatbox = _send_once  # type: ignore[method-assign]
    box4._send_worker()
    assert any(
        args and args[0] == "msg" for _addr, args in sent if isinstance(args, list)
    )

    box5 = OSCChatbox(OSCChatboxConfig())
    box5._text_buffer = "abc"
    box5._send_queue.append(("msg", 0.0))
    box5._send_event.set()
    box5._interrupt_loop(_FakeRecv([InterruptFrame.new(reason="stop"), None]))
    assert box5._text_buffer == ""
    assert list(box5._send_queue) == []

    idle = OSCChatbox(OSCChatboxConfig())
    idle.stop_event.set()
    idle.run(OSCChatboxInputs(), ())

    runner = OSCChatbox(OSCChatboxConfig())
    runner.stop_event.set()
    runner.run(
        OSCChatboxInputs(text=_FakeRecv([None]), interrupt=_FakeRecv([None])), ()
    )
    runner.stop()


def test_osc_face_helper_functions(monkeypatch) -> None:
    class _UDP:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.sent = []

        def send_message(self, address, value):
            self.sent.append((address, value))

    monkeypatch.setattr("src.core.sink.osc_face.SimpleUDPClient", _UDP)
    import src.core.sink.osc_face as osc_face

    monkeypatch.setenv("BOOL_TRUE", " yes ")
    assert osc_face._env_bool("BOOL_TRUE", False) is True
    assert osc_face._env_bool("BOOL_MISSING", True) is True
    assert osc_face._sanitize_env_value(' "abc" ') == "abc"
    assert osc_face._sanitize_env_value(None) == ""
    assert osc_face.lerp(0.0, 10.0, 0.5) == 5.0

    presets = {name: fn() for name, fn in osc_face.EXPRESSION_PRESETS.items()}
    assert "v2/EyeLidLeft" in presets["neutral"]
    base_defaults, keys = osc_face.build_full_targets(osc_face.expression_neutral())
    assert keys
    full = osc_face.preset_to_full("happy", base_defaults)
    assert full["v2/SmileFrownLeft"] > 0
    fallback = osc_face.preset_to_full("missing", base_defaults)
    assert fallback["v2/EyeLidLeft"] == osc_face.expression_neutral()["v2/EyeLidLeft"]

    client = osc_face._OscClient("127.0.0.1", 9000)
    client.send_message("/x", 1.0)
    client.close()

    monkeypatch.setenv("VRCHAT_IP", "1.2.3.4")
    monkeypatch.setenv("VRCHAT_PORT", "9001")
    monkeypatch.setenv("OSC_PREFIX", "/avatar/parameters")
    monkeypatch.setenv("VRCHAT_PARAM_PREFIX", "FT")
    monkeypatch.setenv("HOLD_SECONDS", "1.5")
    monkeypatch.setenv("TRANSITION_SECONDS", "0.5")
    monkeypatch.setenv("TRANSITION_STEPS", "5")
    monkeypatch.setenv("SEND_GAP_SECONDS", "0.0")
    monkeypatch.setenv("OSC_FACE_TEXT_FLUSH_MS", "10")
    monkeypatch.setenv("EMOTION_LLM_ENABLE", "false")
    monkeypatch.setenv("EMOTION_LLM_TIMEOUT_S", "2.0")
    monkeypatch.setenv("EMOTION_LLM_MODEL", '"kimi"')
    monkeypatch.setenv("FUSION_MODE", "bad-mode")
    monkeypatch.setenv("FORCE_LLM", "false")
    monkeypatch.setenv("DEBUG_PRINT_MODEL_REPR", "true")

    face = osc_face.OSCFace()
    assert face.host == "1.2.3.4"
    assert face.port == 9001
    assert face.fusion_mode == "rule_first"
    assert face._build_param_path("JawOpen") == "FT/JawOpen"
