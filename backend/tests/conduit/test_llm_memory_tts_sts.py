from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.frames import (
    AudioFrame,
    EOS,
    InterruptFrame,
    MessageFrame,
    TextFrame,
    ToolCall,
    ToolDef,
)


class _FakeRecv:
    def __init__(self, items):
        self._items = list(items)
        self._iter = iter(self._items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def _audio_frame() -> AudioFrame:
    return AudioFrame.new(
        data=np.zeros((1, 160), dtype=np.float32),
        sample_rate=16000,
        channels=1,
    )


def test_llm_run_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.lib.llm.llm as llm_mod

    class _Function:
        def __init__(self, name: str | None = None, arguments: str | None = None):
            self.name = name
            self.arguments = arguments

    class _ToolCallDelta:
        def __init__(
            self,
            index: int,
            id_: str = "",
            name: str | None = None,
            arguments: str | None = None,
        ):
            self.index = index
            self.id = id_
            self.function = _Function(name, arguments)

    class _Delta:
        def __init__(self, content: str | None = None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _Choice:
        def __init__(self, delta=None, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    class _Chunk:
        def __init__(self, choices):
            self.choices = choices

    monkeypatch.setattr(llm_mod, "ModelResponseStream", _Chunk)

    captured_kwargs = {}

    def fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            object(),
            _Chunk([]),
            _Chunk(
                [
                    _Choice(
                        _Delta(tool_calls=[_ToolCallDelta(0, "id-", "lookup", '{"a":')])
                    )
                ]
            ),
            _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, "1", None, "1}")]))]),
            _Chunk([_Choice(_Delta(content="Hello "))]),
            _Chunk([_Choice(_Delta(content="world"))]),
            _Chunk([_Choice(_Delta(), finish_reason="stop")]),
        ]

    monkeypatch.setattr(llm_mod, "completion", fake_completion)

    llm = llm_mod.LLM(
        llm_mod.LLMConfig(model="demo", top_p=0.8, temperature=0.1, max_tokens=32)
    )
    token_out = []
    text_out = []
    tool_out = []
    eos_out = []
    messages = [
        MessageFrame.new(role="user", content="hi"),
        MessageFrame.new(
            role="assistant",
            content="tool pending",
            tool_calls=[ToolCall.new(call_id="c1", name="old", arguments="{}")],
        ),
        MessageFrame.new(role="tool", content="done", tool_call_id="c1"),
    ]
    llm.run(
        llm_mod.LLMInputs(
            messages=_FakeRecv([messages, None]),
            tools=_FakeRecv(
                [
                    ToolDef.new(
                        name="lookup",
                        description="desc",
                        parameters={"type": "object"},
                        strict=True,
                    ),
                    ToolDef.new(
                        name="ping", description="desc2", parameters={"type": "object"}
                    ),
                    None,
                ]
            ),
            interrupt=_FakeRecv([None, None, None, None, None, None, None]),
        ),
        llm_mod.LLMOutputs(
            token=SimpleNamespace(send=lambda value: token_out.append(value)),
            text=SimpleNamespace(send=lambda value: text_out.append(value)),
            tool_calls=SimpleNamespace(send=lambda value: tool_out.append(value)),
            eos=SimpleNamespace(send=lambda value: eos_out.append(value)),
        ),
    )
    assert captured_kwargs["model"] == "demo"
    assert captured_kwargs["tools"][0]["function"]["strict"] is True
    assert captured_kwargs["tool_choice"] == "auto"
    assert [frame.get() for frame in token_out[:-1]] == ["Hello ", "world"]
    assert token_out[-1] is EOS.END
    assert text_out[0].get() == "Hello world"
    assert tool_out[0].call_id == "id-1"
    assert tool_out[0].arguments == '{"a":1}'
    assert eos_out[-1] is EOS.END

    def interrupt_completion(**kwargs):
        return [
            _Chunk([_Choice(_Delta(content="Hi"))]),
            _Chunk([_Choice(_Delta(content=" ignored"))]),
        ]

    monkeypatch.setattr(llm_mod, "completion", interrupt_completion)
    interrupt_tokens = []
    interrupt_text = []
    interrupt_eos = []
    llm.run(
        llm_mod.LLMInputs(
            messages=_FakeRecv([[MessageFrame.new(role="user", content="hi")], None]),
            interrupt=_FakeRecv([None, InterruptFrame.new(reason="stop"), None]),
        ),
        llm_mod.LLMOutputs(
            token=SimpleNamespace(send=lambda value: interrupt_tokens.append(value)),
            text=SimpleNamespace(send=lambda value: interrupt_text.append(value)),
            tool_calls=SimpleNamespace(send=lambda value: tool_out.append(value)),
            eos=SimpleNamespace(send=lambda value: interrupt_eos.append(value)),
        ),
    )
    assert interrupt_text[0].get() == "Hi"
    assert interrupt_tokens[-1] is EOS.END
    assert interrupt_eos[-1] is EOS.END


def test_llm_setup_warmup_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.lib.llm.llm as llm_mod

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("warmup failed")
        return object()

    monkeypatch.setattr(llm_mod, "completion", fake_completion)

    llm = llm_mod.LLM(llm_mod.LLMConfig(model="warmup-model"))
    llm.setup(
        llm_mod.LLMOutputs(
            token=SimpleNamespace(send=lambda value: None),
            text=None,
            tool_calls=None,
            eos=None,
        )
    )
    llm.setup(
        llm_mod.LLMOutputs(
            token=SimpleNamespace(send=lambda value: None),
            text=None,
            tool_calls=None,
            eos=None,
        )
    )

    assert calls == []


def test_mem0_helpers_and_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import src.lib.llm.memory as memory_mod

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("VECTOR_KEY", "vector-key")
    monkeypatch.setattr(memory_mod.Path, "home", lambda: tmp_path)

    assert memory_mod._require_env("OPENAI_API_KEY") == "openai-key"
    with pytest.raises(ValueError, match="must be set"):
        memory_mod._require_env("MISSING_ENV")

    cfg = memory_mod.Mem0Config(
        llm_base_url="http://llm",
        embedder_base_url="http://embed",
        vector_store=memory_mod.Mem0VectorStoreConfig(
            host="localhost",
            port=1234,
            api_key_env_var="VECTOR_KEY",
        ),
    )
    built = memory_mod._build_mem0_config(cfg)
    assert built["llm"]["config"]["openai_base_url"] == "http://llm"
    assert built["embedder"]["config"]["openai_base_url"] == "http://embed"
    assert built["vector_store"]["config"]["host"] == "localhost"
    assert built["vector_store"]["config"]["api_key"] == "vector-key"

    url_cfg = memory_mod.Mem0Config(
        vector_store=memory_mod.Mem0VectorStoreConfig(url="http://qdrant")
    )
    assert (
        memory_mod._build_mem0_config(url_cfg)["vector_store"]["config"]["url"]
        == "http://qdrant"
    )

    default_cfg = memory_mod.Mem0Config(
        vector_store=memory_mod.Mem0VectorStoreConfig(path=None)
    )
    default_path = memory_mod._build_mem0_config(default_cfg)["vector_store"]["config"][
        "path"
    ]
    assert default_path.endswith("qdrant")

    assert memory_mod._format_memory_prefix([]) == ""
    assert memory_mod._format_memory_prefix(
        [{"memory": "A"}, {"memory": ""}, {"memory": "B"}]
    ) == ("[Relevant memories]\n- A\n- B\n[End of memories]")

    closed = []
    fake_mem = SimpleNamespace(
        vector_store=SimpleNamespace(
            client=SimpleNamespace(close=lambda: closed.append("vector"))
        ),
        _telemetry_vector_store=SimpleNamespace(
            client=SimpleNamespace(close=lambda: closed.append("telemetry"))
        ),
    )
    memory_mod._close_memory(fake_mem)
    assert closed == ["vector", "telemetry"]

    created = []

    class _MemoryFactory:
        @staticmethod
        def from_config(config):
            obj = SimpleNamespace(config=config)
            created.append(obj)
            return obj

    memory_mod._memory_instance = None
    memory_mod._memory_config_hash = None
    close_calls = []
    monkeypatch.setattr(memory_mod, "Memory", _MemoryFactory)
    monkeypatch.setattr(
        memory_mod, "_close_memory", lambda mem: close_calls.append(mem)
    )
    first = memory_mod._get_or_create_memory(memory_mod.Mem0Config(user_id="u1"))
    second = memory_mod._get_or_create_memory(memory_mod.Mem0Config(user_id="u1"))
    third = memory_mod._get_or_create_memory(memory_mod.Mem0Config(user_id="u2"))
    assert first is second
    assert third is not first
    assert close_calls == [first]

    search_calls = []
    add_calls = []

    class _MemoryImpl:
        def search(self, **kwargs):
            search_calls.append(kwargs)
            return {"results": [{"memory": "keep context"}]}

        def add(self, messages, user_id):
            add_calls.append((messages, user_id))

    monkeypatch.setattr(
        memory_mod, "_get_or_create_memory", lambda config: _MemoryImpl()
    )
    mem = memory_mod.Mem0(memory_mod.Mem0Config(last_k=1, memory_limit=2))
    assert mem._query_memory([]) == ""
    assert "keep context" in mem._query_memory(
        [MessageFrame.new(role="user", content="hello")]
    )
    mem._memory.search = lambda **kwargs: [{"memory": "list result"}]
    assert "list result" in mem._query_memory(
        [MessageFrame.new(role="assistant", content="reply")]
    )
    mem._memory.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    assert mem._query_memory([MessageFrame.new(role="user", content="hello")]) == ""
    mem._memory.add = lambda messages, user_id: add_calls.append((messages, user_id))
    mem._update_memory([MessageFrame.new(role="user", content="x")])
    mem._memory.add = lambda messages, user_id: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    mem._update_memory([MessageFrame.new(role="assistant", content="y")])

    update_threads = []

    class _Thread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            update_threads.append(self.args[0])
            self.target(*self.args)

    monkeypatch.setattr(memory_mod.threading, "Thread", _Thread)
    monkeypatch.setattr(mem, "_query_memory", lambda turns: "prefix")
    updated = []
    monkeypatch.setattr(
        mem, "_update_memory", lambda new_messages: updated.append(new_messages)
    )
    prefixes = []
    frames = [
        [],
        [
            MessageFrame.new(role="system", content="ignore"),
            MessageFrame.new(role="user", content="u1"),
            MessageFrame.new(role="assistant", content="a1"),
        ],
        [
            MessageFrame.new(role="system", content="ignore"),
            MessageFrame.new(role="user", content="u1"),
            MessageFrame.new(role="assistant", content="a1"),
        ],
        None,
    ]
    mem.run(
        memory_mod.Mem0Inputs(messages=_FakeRecv(frames)),
        memory_mod.Mem0Outputs(
            memory_prefix=SimpleNamespace(send=lambda value: prefixes.append(value))
        ),
    )
    assert [frame.get() for frame in prefixes] == ["prefix", "prefix"]
    assert len(updated) == 1
    assert [msg.content for msg in updated[0]] == ["u1", "a1"]


def test_tts_worker_and_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.lib.audio.tts as tts_mod

    # litellm (imported by earlier tests) loads .env via dotenv, which sets
    # INWORLD_API_KEY.  Clear it so the "must be set" path is exercised.
    monkeypatch.delenv("INWORLD_API_KEY", raising=False)

    tts = tts_mod.TTS(tts_mod.TTSConfig())
    tts._task_queue.put((0, "hello"))
    with pytest.raises(ValueError, match="must be set"):
        tts._worker(
            tts_mod.TTSOutputs(
                audio=SimpleNamespace(send=lambda value: None),
                text=SimpleNamespace(send=lambda value: None),
            )
        )

    monkeypatch.setenv("INWORLD_API_KEY", "abc12345token")
    ok_tts = tts_mod.TTS(tts_mod.TTSConfig())
    audio_out = []
    text_out = []

    class _Response:
        def __init__(self, fail: bool = False):
            self.fail = fail

        def raise_for_status(self):
            if self.fail:
                raise RuntimeError("bad status")

        def iter_lines(self):
            yield b""
            short = base64.b64encode(b"short")
            yield json.dumps(
                {"result": {"audioContent": short.decode("utf-8")}}
            ).encode("utf-8")
            raw = base64.b64encode(b"R" * 60)
            yield json.dumps({"result": {"audioContent": raw.decode("utf-8")}}).encode(
                "utf-8"
            )

    request_calls = {"count": 0}

    def fake_post(*args, **kwargs):
        request_calls["count"] += 1
        if request_calls["count"] == 1:
            return _Response()
        raise RuntimeError("network")

    monkeypatch.setattr(tts_mod.requests, "post", fake_post)
    sequence = [(1, "stale"), (0, "speak"), (0, "boom")]
    calls = {"count": 0}

    def fake_get(timeout: float):
        idx = calls["count"]
        calls["count"] += 1
        if idx < len(sequence):
            return sequence[idx]
        ok_tts.stop_event.set()
        raise tts_mod.Empty

    monkeypatch.setattr(ok_tts._task_queue, "get", fake_get)
    ok_tts._worker(
        tts_mod.TTSOutputs(
            audio=SimpleNamespace(send=lambda value: audio_out.append(value)),
            text=SimpleNamespace(send=lambda value: text_out.append(value)),
        )
    )
    assert len(audio_out) == 1
    assert text_out[0].get() == "speak"

    mismatch_tts = tts_mod.TTS(tts_mod.TTSConfig())
    monkeypatch.setenv("INWORLD_API_KEY", "abc12345token")
    mismatch_text = []

    class _MismatchResponse:
        def raise_for_status(self):
            return

        def iter_lines(self):
            payload = json.dumps(
                {
                    "result": {
                        "audioContent": base64.b64encode(b"R" * 60).decode("utf-8")
                    }
                }
            ).encode("utf-8")
            yield payload
            yield payload

    monkeypatch.setattr(
        tts_mod.requests, "post", lambda *args, **kwargs: _MismatchResponse()
    )
    monkeypatch.setattr(
        mismatch_tts._task_queue,
        "get",
        lambda timeout: mismatch_tts.stop_event.set() or (0, "x"),
    )

    def flip_generation(value):
        mismatch_tts._generation = 1

    mismatch_tts._worker(
        tts_mod.TTSOutputs(
            audio=SimpleNamespace(send=flip_generation),
            text=SimpleNamespace(send=lambda value: mismatch_text.append(value)),
        )
    )
    assert mismatch_text == []

    run_tts = tts_mod.TTS(tts_mod.TTSConfig())
    puts = []

    class _Filter:
        def __init__(self):
            self.calls = []

        def feed(self, token: str, force: bool = False) -> str:
            self.calls.append((token, force))
            if force:
                return "forced text"
            return "stream text"

    monkeypatch.setattr(tts_mod, "StreamFilter", _Filter)
    run_tts._stream_filter = _Filter()

    class _Thread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            self.joined = False

        def start(self):
            self.started = True
            if self.target.__name__ == "handle_interrupts":
                self.target(*self.args)

        def join(self, timeout=None):
            self.joined = True
            self.timeout = timeout

        def is_alive(self):
            return False

    monkeypatch.setattr(tts_mod.threading, "Thread", _Thread)
    monkeypatch.setattr(run_tts._task_queue, "put", lambda item: puts.append(item))
    monkeypatch.setattr(run_tts._task_queue, "empty", lambda: False)
    monkeypatch.setattr(
        run_tts._task_queue,
        "get_nowait",
        lambda: (_ for _ in ()).throw(tts_mod.Empty),
    )
    run_tts.run(
        tts_mod.TTSInputs(
            text=_FakeRecv([TextFrame.new(text="hello"), EOS.END, None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="stop"), None]),
        ),
        tts_mod.TTSOutputs(
            audio=SimpleNamespace(send=lambda value: None),
            text=SimpleNamespace(send=lambda value: None),
        ),
    )
    assert run_tts._generation == 1
    assert puts == [(1, "stream text"), (1, "forced text")]


def test_sts_stop_run_and_send_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import websockets.exceptions
    import src.lib.audio.sts as sts_mod

    class _Closed(Exception):
        pass

    monkeypatch.setattr(websockets.exceptions, "ConnectionClosed", _Closed)

    sts = sts_mod.STS(sts_mod.STSConfig())
    closed = {"count": 0}
    sts._ws = SimpleNamespace(
        close=lambda: closed.__setitem__("count", closed["count"] + 1)
    )
    sts.stop()
    assert closed["count"] == 1

    sts2 = sts_mod.STS(sts_mod.STSConfig())
    sts2._ws = SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    sts2.stop()

    ws_messages = []
    sender = sts_mod.STS(sts_mod.STSConfig())
    sender._send_loop(
        SimpleNamespace(send=lambda payload: ws_messages.append(json.loads(payload))),
        _FakeRecv([_audio_frame(), None]),
    )
    assert ws_messages[0]["type"] == "input_audio_buffer.append"

    def raising_send(payload):
        raise _Closed()

    sender._send_loop(
        SimpleNamespace(send=raising_send), _FakeRecv([_audio_frame(), None])
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="must be set"):
        sts_mod.STS(sts_mod.STSConfig()).run(
            sts_mod.STSInputs(audio=_FakeRecv([None])),
            sts_mod.STSOutputs(audio=SimpleNamespace(send=lambda value: None)),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    audio_out = []

    class _WS:
        def __init__(self, owner):
            self.owner = owner
            self.sent = []
            self._messages = [
                json.dumps(
                    {
                        "type": "response.audio.delta",
                        "delta": base64.b64encode(b"\x00" * 8).decode("utf-8"),
                    }
                ),
                json.dumps({"type": "response.created"}),
            ]
            self._index = 0

        def send(self, payload):
            self.sent.append(json.loads(payload))

        def close(self):
            return

        def __iter__(self):
            return self

        def __next__(self):
            if self._index >= len(self._messages):
                raise StopIteration
            value = self._messages[self._index]
            self._index += 1
            if self._index > 1:
                self.owner.stop_event.set()
            return value

    class _ConnectCtx:
        def __init__(self, owner):
            self.ws = _WS(owner)

        def __enter__(self):
            return self.ws

        def __exit__(self, *args):
            return False

    class _Thread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            self.target(*self.args)

        def join(self, timeout=None):
            self.timeout = timeout

    run_sts = sts_mod.STS(sts_mod.STSConfig())
    monkeypatch.setattr(
        sts_mod, "connect", lambda url, additional_headers: _ConnectCtx(run_sts)
    )
    monkeypatch.setattr(sts_mod.threading, "Thread", _Thread)
    run_sts.run(
        sts_mod.STSInputs(
            audio=_FakeRecv([_audio_frame(), None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="clear"), None]),
        ),
        sts_mod.STSOutputs(
            audio=SimpleNamespace(send=lambda value: audio_out.append(value))
        ),
    )
    assert len(audio_out) == 1
    assert run_sts._ws is None
