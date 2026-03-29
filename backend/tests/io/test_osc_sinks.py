from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.core.frames import InterruptFrame, TextFrame


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def test_osc_chatbox_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []

    class _UDP:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def send_message(self, address, args):
            sent.append((address, args))

    monkeypatch.setattr("src.core.sink.osc_chatbox.SimpleUDPClient", _UDP)

    import src.core.sink.osc_chatbox as osc_chatbox

    client = osc_chatbox._OscClient("127.0.0.1", 9000)
    client.send_message("/chatbox/input", ["x", True, False])
    client.close()
    assert sent[-1] == ("/chatbox/input", ["x", True, False])

    box = osc_chatbox.OSCChatbox(osc_chatbox.OSCChatboxConfig(max_chars=6))
    assert box._split_text("one two") == ["one", "two"]
    assert box._split_text("abc de fghij")[0] == "abc de"
    box._enqueue_text("   ")
    assert list(box._send_queue) == []

    disabled_flush = osc_chatbox.OSCChatbox(
        osc_chatbox.OSCChatboxConfig(text_flush_ms=0)
    )
    disabled_flush._text_flush_monitor()

    empty_box = osc_chatbox.OSCChatbox(osc_chatbox.OSCChatboxConfig(text_flush_ms=1))

    def empty_sleep(_seconds: float) -> None:
        empty_box.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_chatbox.time.sleep", empty_sleep)
    empty_box._text_flush_monitor()

    flushed = []
    flush_box = osc_chatbox.OSCChatbox(osc_chatbox.OSCChatboxConfig(text_flush_ms=1))
    flush_box._text_buffer = "idle text"
    flush_box._last_text_time = 0.0
    monkeypatch.setattr("src.core.sink.osc_chatbox.time.monotonic", lambda: 1.0)

    def fake_sleep(_seconds: float) -> None:
        flush_box.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_chatbox.time.sleep", fake_sleep)
    monkeypatch.setattr(flush_box, "_enqueue_text", lambda text: flushed.append(text))
    flush_box._text_flush_monitor()
    assert flushed == ["idle text"]

    continue_box = osc_chatbox.OSCChatbox(
        osc_chatbox.OSCChatboxConfig(text_flush_ms=1000)
    )
    continue_box._text_buffer = "busy"
    continue_box._last_text_time = 0.95
    continue_calls = {"count": 0}
    monkeypatch.setattr("src.core.sink.osc_chatbox.time.monotonic", lambda: 1.0)

    def continue_sleep(_seconds: float) -> None:
        continue_calls["count"] += 1
        continue_box.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_chatbox.time.sleep", continue_sleep)
    continue_box._text_flush_monitor()
    assert continue_box._text_buffer == "busy"

    worker_box = osc_chatbox.OSCChatbox(
        osc_chatbox.OSCChatboxConfig(clear_on_last=True)
    )
    sleeps = []
    monkeypatch.setattr(
        "src.core.sink.osc_chatbox.time.sleep", lambda seconds: sleeps.append(seconds)
    )

    def fake_send(text: str, *, reset: bool = False) -> None:
        sent.append(("worker", [text, reset]))
        if reset:
            worker_box.stop_event.set()

    worker_box._send_chatbox = fake_send  # type: ignore[method-assign]
    worker_box._send_queue.append(("msg", 0.5))
    worker_box._send_event.set()
    worker_box._send_worker()
    assert 0.5 in sleeps
    assert ("worker", ["msg", False]) in sent
    assert ("worker", ["", True]) in sent

    idle_box = osc_chatbox.OSCChatbox(osc_chatbox.OSCChatboxConfig())
    sleep_calls = {"count": 0}

    def idle_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        idle_box.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_chatbox.time.sleep", idle_sleep)
    idle_box.run(osc_chatbox.OSCChatboxInputs(), ())
    assert sleep_calls["count"] == 1


def test_osc_face_branches(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    sent = []

    class _UDP:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def send_message(self, address, value):
            sent.append((address, value))

    monkeypatch.setattr("src.core.sink.osc_face.SimpleUDPClient", _UDP)
    import src.core.sink.osc_face as osc_face

    cases = [
        ("expression: happy", "happy", True),
        ("terrified", "scared", True),
        ("embarrassed", "shy", True),
        ("sneer", "smirk", True),
        ("very sleepy", "sleepy", True),
        ("adorable", "cute", True),
        ("thank you", "happy", True),
        ("heartbroken", "sad", True),
        ("furious", "angry", True),
        ("oh my god", "surprised", True),
        ("what now?", "thinking", False),
        ("plain text", "neutral", False),
    ]

    for text, expected, strong in cases:
        face = osc_face.OSCFace(osc_face.OSCFaceConfig(debug_print_model_repr=False))
        expr, hits, is_strong = face._select_expression_rule(text)
        assert expr == expected
        assert is_strong is strong
        assert isinstance(hits, list)

    disabled = osc_face.OSCFace(
        osc_face.OSCFaceConfig(emotion_llm_enable=False, debug_print_model_repr=False)
    )
    assert disabled._select_expression_llm("x")[1].startswith("LLM disabled")

    no_url = osc_face.OSCFace(
        osc_face.OSCFaceConfig(emotion_llm_enable=True, debug_print_model_repr=False)
    )
    assert no_url._select_expression_llm("x")[1] == "Missing EMOTION_LLM_URL"

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    missing_key = osc_face.OSCFace(
        osc_face.OSCFaceConfig(
            emotion_llm_enable=True,
            emotion_llm_url="http://llm",
            debug_print_model_repr=False,
        )
    )
    assert missing_key._select_expression_llm("x")[1] == "Missing GROQ_API_KEY"

    monkeypatch.setenv("GROQ_API_KEY", "secret")

    class _Response:
        def __init__(self, status_code: int, content: str, text: str = "bad"):
            self.status_code = status_code
            self._content = content
            self.text = text

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    http_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(
            emotion_llm_enable=True,
            emotion_llm_url="http://llm",
            debug_print_model_repr=False,
        )
    )
    monkeypatch.setattr(
        "src.core.sink.osc_face.requests.post",
        lambda *args, **kwargs: _Response(
            500, json.dumps({"expression": "happy"}), "oops"
        ),
    )
    assert http_face._select_expression_llm("x")[1] == "LLM HTTP 500: oops"

    invalid_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(
            emotion_llm_enable=True,
            emotion_llm_url="http://llm",
            debug_print_model_repr=False,
        )
    )
    monkeypatch.setattr(
        "src.core.sink.osc_face.requests.post",
        lambda *args, **kwargs: _Response(200, json.dumps({"expression": "unknown"})),
    )
    assert invalid_face._select_expression_llm("x")[1].startswith("LLM invalid output:")

    monkeypatch.setattr(
        "src.core.sink.osc_face.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert invalid_face._select_expression_llm("x")[1].startswith("LLM error:")

    valid_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(
            emotion_llm_enable=True,
            emotion_llm_url="http://llm",
            debug_print_model_repr=False,
            emotion_llm_model="demo",
        )
    )
    monkeypatch.setattr(
        "src.core.sink.osc_face.requests.post",
        lambda *args, **kwargs: _Response(200, json.dumps({"expression": "cute"})),
    )
    assert valid_face._select_expression_llm("x") == (
        "cute",
        'LLM ok: {"expression": "cute"}',
    )

    llm_first = osc_face.OSCFace(
        osc_face.OSCFaceConfig(
            fusion_mode="llm_first",
            emotion_llm_enable=True,
            debug_print_model_repr=False,
        )
    )
    monkeypatch.setattr(
        llm_first, "_select_expression_llm", lambda text: ("happy", "ok")
    )
    expr, dbg = llm_first._fuse_expression("plain text")
    assert expr == "happy"
    assert dbg["final_reason"] == "llm_first"

    override_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(emotion_llm_enable=True, debug_print_model_repr=False)
    )
    monkeypatch.setattr(
        override_face, "_select_expression_llm", lambda text: ("sad", "ok")
    )
    expr, dbg = override_face._fuse_expression("plain text")
    assert expr == "sad"
    assert dbg["final_reason"] == "rule_weak->llm_override"

    fallback_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(emotion_llm_enable=True, debug_print_model_repr=False)
    )
    monkeypatch.setattr(
        fallback_face, "_select_expression_llm", lambda text: (None, "bad")
    )
    expr, dbg = fallback_face._fuse_expression("what now?")
    assert expr == "thinking"
    assert dbg["final_reason"] == "rule_weak->llm_failed_fallback_rule"

    strong_face = osc_face.OSCFace(osc_face.OSCFaceConfig(debug_print_model_repr=False))
    expr, dbg = strong_face._fuse_expression("furious")
    assert expr == "angry"
    assert dbg["final_reason"] == "rule_strong_or_llm_off"

    sender_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(
            send_gap_seconds=0.25, transition_steps=2, transition_seconds=0.4
        )
    )
    sleeps = []
    monkeypatch.setattr(
        "src.core.sink.osc_face.time.sleep", lambda seconds: sleeps.append(seconds)
    )
    sender_face._send_params({"JawOpen": 0.5})
    assert sent[-1][0].endswith("/FT/JawOpen")
    assert 0.25 in sleeps

    transition_frames = []
    sender_face._keys = ["JawOpen", "MouthClosed"]
    sender_face._send_params = lambda params: transition_frames.append(params.copy())  # type: ignore[method-assign]
    result = sender_face._transition(
        {"JawOpen": 0.0, "MouthClosed": 0.0},
        {"JawOpen": 1.0, "MouthClosed": 0.5},
    )
    assert result == {"JawOpen": 1.0, "MouthClosed": 0.5}
    assert transition_frames[-1]["JawOpen"] == 1.0

    sender_face._enqueue_text("   ")
    assert list(sender_face._expr_queue) == []
    sender_face._enqueue_text(" hello ")
    assert list(sender_face._expr_queue) == ["hello"]
    sender_face._text_buffer = "pending"
    flushed = []
    monkeypatch.setattr(sender_face, "_enqueue_text", lambda text: flushed.append(text))
    sender_face._flush_text_buffer()
    assert flushed == ["pending"]

    text_face = osc_face.OSCFace(osc_face.OSCFaceConfig(debug_print_model_repr=False))
    text_enqueued = []
    monkeypatch.setattr(
        text_face, "_enqueue_text", lambda text: text_enqueued.append(text)
    )
    monotonic_values = iter([1.0, 2.0])
    monkeypatch.setattr(
        "src.core.sink.osc_face.time.monotonic", lambda: next(monotonic_values)
    )
    text_face._text_loop(
        _FakeRecv(
            [
                TextFrame.new(text="a"),
                TextFrame.new(text=osc_face.GENERATE_END_FLAG),
                None,
            ]
        )
    )
    assert text_enqueued == ["a"]

    no_flush_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(text_flush_ms=0, debug_print_model_repr=False)
    )
    no_flush_face._text_flush_monitor()

    empty_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(text_flush_ms=1, debug_print_model_repr=False)
    )

    def empty_face_sleep(_seconds: float) -> None:
        empty_face.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_face.time.sleep", empty_face_sleep)
    empty_face._text_flush_monitor()

    flush_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(text_flush_ms=1, debug_print_model_repr=False)
    )
    flush_face._text_buffer = "later"
    flush_face._last_text_time = 0.0
    flushed_idle = []
    monkeypatch.setattr("src.core.sink.osc_face.time.monotonic", lambda: 1.0)

    def stop_sleep(_seconds: float) -> None:
        flush_face.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_face.time.sleep", stop_sleep)
    monkeypatch.setattr(
        flush_face, "_enqueue_text", lambda text: flushed_idle.append(text)
    )
    flush_face._text_flush_monitor()
    assert flushed_idle == ["later"]

    continue_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(text_flush_ms=1000, debug_print_model_repr=False)
    )
    continue_face._text_buffer = "busy"
    continue_face._last_text_time = 0.95
    monkeypatch.setattr("src.core.sink.osc_face.time.monotonic", lambda: 1.0)

    def continue_face_sleep(_seconds: float) -> None:
        continue_face.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_face.time.sleep", continue_face_sleep)
    continue_face._text_flush_monitor()
    assert continue_face._text_buffer == "busy"

    interrupt_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(debug_print_model_repr=False)
    )
    interrupt_face._text_buffer = "x"
    interrupt_face._expr_queue.append("queued")
    interrupt_face._expr_event.set()
    interrupt_face._interrupt_loop(_FakeRecv([InterruptFrame.new(reason="stop"), None]))
    assert interrupt_face._text_buffer == ""
    assert list(interrupt_face._expr_queue) == []
    assert interrupt_face._expr_event.is_set() is False

    worker_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(hold_seconds=0.2, debug_print_model_repr=True)
    )
    monkeypatch.setattr(
        worker_face,
        "_fuse_expression",
        lambda text: (
            "happy",
            {
                "rule_expr": "happy",
                "rule_hits": "yay",
                "rule_strong": "1",
                "llm_expr": "(none)",
                "llm_detail": "(none)",
                "final_reason": "rule",
            },
        ),
    )

    def fake_transition(current, target):
        worker_face.stop_event.set()
        return target

    monkeypatch.setattr(worker_face, "_transition", fake_transition)
    worker_face._expr_queue.append("hello")
    worker_face._expr_event.set()
    sleep_values = []
    monkeypatch.setattr(
        "src.core.sink.osc_face.time.sleep",
        lambda seconds: sleep_values.append(seconds),
    )
    worker_face._expression_worker()
    printed = capsys.readouterr().out
    assert "[OSCFace DEBUG]" in printed
    assert 0.2 in sleep_values

    stop_face = osc_face.OSCFace(osc_face.OSCFaceConfig(debug_print_model_repr=False))
    closed = {"value": False}
    stop_face._client = SimpleNamespace(close=lambda: closed.__setitem__("value", True))
    stop_face.stop()
    assert closed["value"] is True
    assert stop_face.stop_event.is_set() is True

    run_face = osc_face.OSCFace(osc_face.OSCFaceConfig(debug_print_model_repr=True))
    sent_params = []
    run_face._send_params = lambda params: sent_params.append(params.copy())  # type: ignore[method-assign]
    sleep_counter = {"count": 0}

    def run_sleep(seconds: float) -> None:
        sleep_counter["count"] += 1
        if seconds == 0.1:
            run_face.stop_event.set()

    monkeypatch.setattr("src.core.sink.osc_face.time.sleep", run_sleep)
    run_face.run(osc_face.OSCFaceInputs(), ())
    config_output = capsys.readouterr().out
    assert "[OSCFace CONFIG]" in config_output
    assert sent_params

    threaded_face = osc_face.OSCFace(
        osc_face.OSCFaceConfig(debug_print_model_repr=False)
    )
    calls = []
    threaded_face._send_params = lambda params: calls.append("send")  # type: ignore[method-assign]
    threaded_face._text_loop = lambda recv: calls.append("text")  # type: ignore[method-assign]
    threaded_face._text_flush_monitor = lambda: calls.append("flush")  # type: ignore[method-assign]
    threaded_face._expression_worker = lambda: calls.append("worker")  # type: ignore[method-assign]
    threaded_face._interrupt_loop = lambda recv: calls.append("interrupt")  # type: ignore[method-assign]

    class _FakeThread:
        def __init__(self, target, args=()):
            self.target = target
            self.args = args
            self.daemon = False
            self.started = False
            self.joined = False

        def start(self) -> None:
            self.started = True
            self.target(*self.args)

        def join(self) -> None:
            self.joined = True

    monkeypatch.setattr("src.core.sink.osc_face.threading.Thread", _FakeThread)
    monkeypatch.setattr("src.core.sink.osc_face.time.sleep", lambda seconds: None)
    threaded_face.run(
        osc_face.OSCFaceInputs(text=_FakeRecv([None]), interrupt=_FakeRecv([None])),
        (),
    )
    assert calls == ["send", "text", "flush", "worker", "interrupt"]
