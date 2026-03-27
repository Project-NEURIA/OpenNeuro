from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.frames import EOS, GoalFrame, InterruptFrame, TextFrame


class _FakeRecv:
    def __init__(self, items):
        self._items = list(items)

    def __call__(self, *args, **kwargs):
        items = list(self._items)

        def _gen():
            while items:
                yield items.pop(0)
            while True:
                yield None

        return _gen()


class _OneShotStopEvent:
    def __init__(self):
        self._flag = False

    def is_set(self):
        return self._flag

    def set(self):
        self._flag = True

    def wait(self, _timeout=None):
        return None


class _FakeFuture:
    def __init__(self, value=None):
        self.value = value
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        return True

    def result(self):
        return self.value


def exercise_discord_qwen_component_segmenter_and_pose_branches(
    monkeypatch, tmp_path: Path
) -> None:
    import src.core.conduit.discord as discord_mod
    import src.core.conduit.pose_renderer_3d as pose3d_mod
    import src.core.conduit.qwen_tts.component as comp_mod
    import src.core.conduit.yolo.object_segmenter as seg_mod

    discord_mod._discord_bot = None
    discord_mod._discord_loop = None
    discord_mod._discord_thread = None
    discord_mod._discord_running = False
    discord_mod._active_discord_io = None
    discord_mod._voice_clients.clear()
    discord_mod._rings.clear()
    discord_mod._buffer.clear()
    discord_mod._playback_tasks.clear()

    monkeypatch.setattr(discord_mod.os, "getenv", lambda _key: "token")

    class _LazyThread:
        def __init__(self, target, args=(), daemon=False, **kwargs):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            return None

    class _LazyEvent:
        def wait(self, _timeout):
            discord_mod._discord_bot = object()
            return None

    monkeypatch.setattr(discord_mod.threading, "Thread", _LazyThread)
    monkeypatch.setattr(discord_mod.threading, "Event", lambda: _LazyEvent())
    io_lazy = discord_mod.DiscordIO(discord_mod.DiscordConfig())
    assert io_lazy.token == "token"

    class _SlashBot:
        def __init__(self):
            self.events = {}
            self.commands = {}
            self.user = "bot-user"

        def event(self, fn):
            self.events[fn.__name__] = fn
            return fn

        def slash_command(self, name, guild_ids=None):
            def decorator(fn):
                self.commands[name] = fn
                return fn

            return decorator

    class _Member:
        def __init__(self, voice):
            self.voice = voice

    class _Ctx:
        def __init__(self, author, guild):
            self.author = author
            self.guild = guild
            self.responses = []
            self.deferred = False
            self.followup = SimpleNamespace(messages=[])

        async def respond(self, text):
            self.responses.append(text)

        async def defer(self):
            self.deferred = True

    async def _followup_send(self, text):
        self.messages.append(text)

    discord_mod._discord_bot = object()
    discord_mod._discord_running = True
    monkeypatch.setattr(discord_mod.discord, "Member", _Member)
    monkeypatch.setattr(
        discord_mod.DiscordIO, "_ensure_discord_running", lambda self: None
    )

    io = discord_mod.DiscordIO(discord_mod.DiscordConfig(guild_ids=[1]))
    bot = _SlashBot()
    io._register_handlers_for_bot(bot)

    class _BadExistingVC:
        def __init__(self, guild):
            self.guild = guild

        async def disconnect(self, force=True):
            raise RuntimeError("disconnect failed")

        def cleanup(self):
            raise RuntimeError("cleanup failed")

    class _GoodNewVC:
        def __init__(self):
            self._connected = SimpleNamespace(wait=lambda timeout: True)
            self.recording = False

        def is_connected(self):
            return True

        def start_recording(self, sink, callback):
            asyncio.get_running_loop().create_task(callback(sink))

        async def disconnect(self, force=True):
            return None

        def play(self, source):
            self.source = source

    class _Channel:
        def __init__(self, result):
            self.result = result

        async def connect(self, timeout=30.0, reconnect=False):
            return self.result

    class _Guild:
        def __init__(self, gid, voice_client=None):
            self.id = gid
            self.voice_client = voice_client

    existing_guild = _Guild(3)
    existing_guild.voice_client = _BadExistingVC(existing_guild)
    join_ctx = _Ctx(
        author=_Member(voice=SimpleNamespace(channel=_Channel(_GoodNewVC()))),
        guild=existing_guild,
    )
    join_ctx.followup.send = _followup_send.__get__(join_ctx.followup, object)
    async def _sleep(_s):
        return None

    monkeypatch.setattr(discord_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        discord_mod.asyncio,
        "create_task",
        lambda coro: (
            coro.close(),
            SimpleNamespace(done=lambda: True, cancel=lambda: None),
        )[1],
    )
    asyncio.run(bot.commands["join"](join_ctx))
    assert join_ctx.followup.messages == ["Connected"]

    class _BadLeaveVC:
        def __init__(self):
            self.recording = True

        def stop_recording(self):
            raise RuntimeError("stop failed")

        async def disconnect(self, force=True):
            raise RuntimeError("disconnect failed")

    discord_mod._voice_clients[9] = _BadLeaveVC()
    discord_mod._playback_tasks[9] = SimpleNamespace(cancel=lambda: None)
    leave_ctx = _Ctx(author=_Member(voice=None), guild=SimpleNamespace(id=9))
    asyncio.run(bot.commands["leave"](leave_ctx))
    assert leave_ctx.responses == ["Disconnected"]

    comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    comp._voice_paths = {"demo": Path("demo.pt")}
    audio_sent = []

    class _WorkerQueue:
        def __init__(self):
            self.items = [(0, "cancel")]

        def get(self, timeout=None):
            if self.items:
                return self.items.pop(0)
            comp.stop_event.set()
            raise comp_mod.Empty

        def empty(self):
            return True

    class _TopCancelEvent:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

        def set(self):
            self.calls = 99

    comp._tts = SimpleNamespace(
        sample_rate=24000,
        generate_streaming=lambda text, voice_path, language: iter(
            [
                (np.array([0.1], dtype=np.float32), {}),
                (np.array([0.2], dtype=np.float32), {}),
            ]
        ),
    )
    comp._task_queue = _WorkerQueue()
    monkeypatch.setattr(comp_mod.threading, "Event", _TopCancelEvent)
    comp._worker(
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda frame: audio_sent.append(frame)),
            text=SimpleNamespace(send=lambda frame: None),
        )
    )
    assert len(audio_sent) == 1

    comp2 = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    comp2._voice_paths = {"demo": Path("demo.pt")}
    mismatch_audio = []

    class _MismatchQueue:
        def __init__(self):
            self.items = [(0, "run")]

        def get(self, timeout=None):
            if self.items:
                return self.items.pop(0)
            comp2.stop_event.set()
            raise comp_mod.Empty

        def empty(self):
            return True

    def _mismatch_stream(*args, **kwargs):
        yield np.array([0.1], dtype=np.float32), {}
        comp2._generation = 1
        yield np.array([0.2], dtype=np.float32), {}

    comp2._tts = SimpleNamespace(sample_rate=24000, generate_streaming=_mismatch_stream)
    comp2._task_queue = _MismatchQueue()
    comp2._worker(
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda frame: mismatch_audio.append(frame)),
            text=SimpleNamespace(send=lambda frame: None),
        )
    )
    assert len(mismatch_audio) == 1

    run_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    run_comp._load_model = lambda: None
    run_comp._register_voices = lambda: None
    run_comp._worker = lambda outputs: None
    monkeypatch.setattr(
        comp_mod.threading,
        "Thread",
        lambda target, args=(), daemon=False, **kwargs: SimpleNamespace(
            start=lambda: target(*args),
            join=lambda timeout=None: None,
        ),
    )

    class _InterruptQueue:
        def __init__(self):
            self._empty = False

        def empty(self):
            return self._empty is False

        def get_nowait(self):
            self._empty = True
            raise comp_mod.Empty

        def put(self, item):
            return None

    run_comp._task_queue = _InterruptQueue()
    run_comp.run(
        comp_mod.QwenTTSInputs(
            text=_FakeRecv([EOS.END, None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="stop"), None]),
        ),
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda frame: None),
            text=SimpleNamespace(send=lambda frame: None),
        ),
    )

    segmenter = seg_mod.ObjectSegmenter(seg_mod.ObjectSegmenterConfig())
    segmenter._prompts = ["cat"]

    class _Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value)

        def int(self):
            return self

    class _Boxes:
        def __init__(self):
            self.xyxy = _Tensor([[0, 0, 1, 1]])
            self.conf = _Tensor([0.9])
            self.cls = _Tensor([5])
            self.id = None
            self.is_track = False

        def __len__(self):
            return 1

    segmenter._model = SimpleNamespace(
        predict=lambda frame, **kwargs: [SimpleNamespace(boxes=_Boxes(), masks=None)]
    )
    segmenter._tracker_yaml = None
    assert segmenter._infer(np.zeros((2, 2, 3), dtype=np.uint8)) is None

    class _FakeMesh:
        def __init__(self, vertices=None, faces=None, process=False):
            self.vertices = (
                np.asarray(vertices) if vertices is not None else np.zeros((1, 3))
            )
            self.faces = (
                np.asarray(faces)
                if faces is not None
                else np.zeros((1, 3), dtype=np.int32)
            )
            self.transforms = []
            self.bounds = np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
            )

        def apply_transform(self, mat):
            self.transforms.append(mat)

    renderer = object.__new__(pose3d_mod.PoseRenderer3D)
    renderer.config = pose3d_mod.PoseRenderer3DConfig(
        width=4, height=4, camera_distance=2.0, device="cpu"
    )
    renderer._faces = np.array([[0, 1, 2]], dtype=np.int32)
    renderer._scene = SimpleNamespace(nodes=[])
    renderer._scene.add = (
        lambda obj, name=None, pose=None: renderer._scene.nodes.append(
            SimpleNamespace(obj=obj, name=name, pose=pose)
        )
        or renderer._scene.nodes[-1]
    )
    renderer._scene.remove_node = lambda node: renderer._scene.nodes.remove(node)
    renderer._renderer = SimpleNamespace(
        render=lambda scene, flags=None: (np.zeros((4, 4, 4), dtype=np.uint8), None)
    )
    renderer._pyrender = SimpleNamespace(
        MetallicRoughnessMaterial=lambda **kwargs: SimpleNamespace(**kwargs),
        Mesh=SimpleNamespace(
            from_trimesh=lambda mesh, material=None: SimpleNamespace(
                mesh=mesh, material=material
            )
        ),
        PerspectiveCamera=lambda **kwargs: SimpleNamespace(**kwargs),
        RenderFlags=SimpleNamespace(RGBA="rgba"),
    )
    renderer._trimesh = SimpleNamespace(
        Trimesh=_FakeMesh,
        creation=SimpleNamespace(cylinder=lambda **kwargs: _FakeMesh()),
        transformations=SimpleNamespace(
            rotation_matrix=lambda angle, axis: np.eye(4, dtype=np.float32)
        ),
    )
    real_allclose = pose3d_mod.np.allclose
    state = {"forced": False}

    def _branch_allclose(a, b, *args, **kwargs):
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        if (
            not state["forced"]
            and np.array_equal(arr_a, np.array([0.0, 0.0, 1.0]))
            and np.array_equal(arr_b, np.array([0.0, 0.0, 1.0]))
        ):
            state["forced"] = True
            return False
        if (
            state["forced"]
            and np.array_equal(arr_a, np.array([0.0, 0.0, 1.0]))
            and np.array_equal(arr_b, np.array([0.0, 0.0, -1.0]))
        ):
            return True
        return real_allclose(a, b, *args, **kwargs)

    monkeypatch.setattr(pose3d_mod.np, "allclose", _branch_allclose)
    image = renderer._render_mesh(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert image.shape == (4, 4, 3)


def test_discord_cleanup_branches(monkeypatch) -> None:
    import src.core.conduit.discord as discord_mod
    import time

    discord_mod._discord_bot = None
    discord_mod._discord_running = False
    discord_mod._voice_clients.clear()
    discord_mod._playback_tasks.clear()
    discord_mod._rings.clear()
    discord_mod._buffer.clear()
    monkeypatch.setattr(discord_mod.os, "getenv", lambda _key: "token")
    monkeypatch.setattr(discord_mod.discord.Intents, "default", lambda: "intents")

    class _StartupBot:
        def __init__(self, intents):
            time.sleep(0.02)
            self.user = "boot"

        async def start(self, token):
            return None

    class _FakeLoop:
        def run_until_complete(self, coro):
            return asyncio.run(coro)

    monkeypatch.setattr(discord_mod.discord, "Bot", _StartupBot)
    monkeypatch.setattr(discord_mod.asyncio, "new_event_loop", lambda: _FakeLoop())
    monkeypatch.setattr(discord_mod.asyncio, "set_event_loop", lambda loop: None)
    real_register_handlers = discord_mod.DiscordIO._register_handlers_for_bot
    monkeypatch.setattr(
        discord_mod.DiscordIO, "_register_handlers_for_bot", lambda self, bot: None
    )
    discord_mod.DiscordIO(discord_mod.DiscordConfig())

    class _Member:
        def __init__(self, voice):
            self.voice = voice

    class _SlashBot:
        def __init__(self):
            self.events = {}
            self.commands = {}
            self.user = "bot"

        def event(self, fn):
            self.events[fn.__name__] = fn
            return fn

        def slash_command(self, name, guild_ids=None):
            def decorator(fn):
                self.commands[name] = fn
                return fn

            return decorator

    class _Ctx:
        def __init__(self, author, guild):
            self.author = author
            self.guild = guild
            self.responses = []
            self.deferred = False
            self.followup = SimpleNamespace(messages=[])

        async def respond(self, text):
            self.responses.append(text)

        async def defer(self):
            self.deferred = True

    async def _followup_send(self, text):
        self.messages.append(text)

    class _BadExistingVC:
        def __init__(self, guild):
            self.guild = guild

        async def disconnect(self, force=True):
            raise RuntimeError("disconnect failed")

        def cleanup(self):
            raise RuntimeError("cleanup failed")

    class _GoodNewVC:
        def __init__(self):
            self._connected = SimpleNamespace(wait=lambda timeout: True)
            self.recording = False

        def is_connected(self):
            return True

        def start_recording(self, sink, callback):
            return None

        async def disconnect(self, force=True):
            return None

        def play(self, source):
            self.source = source

    class _Channel:
        def __init__(self, result):
            self.result = result

        async def connect(self, timeout=30.0, reconnect=False):
            return self.result

    class _Guild:
        def __init__(self, gid, voice_client=None):
            self.id = gid
            self.voice_client = voice_client

    discord_mod._discord_bot = object()
    discord_mod._discord_running = True
    monkeypatch.setattr(discord_mod.discord, "Member", _Member)
    monkeypatch.setattr(
        discord_mod.DiscordIO, "_ensure_discord_running", lambda self: None
    )
    monkeypatch.setattr(
        discord_mod.DiscordIO, "_register_handlers_for_bot", real_register_handlers
    )
    async def _sleep(_s):
        return None
    monkeypatch.setattr(discord_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        discord_mod.asyncio,
        "create_task",
        lambda coro: (
            coro.close(),
            SimpleNamespace(done=lambda: True, cancel=lambda: None),
        )[1],
    )
    io = discord_mod.DiscordIO(discord_mod.DiscordConfig(guild_ids=[1]))
    bot = _SlashBot()
    io._register_handlers_for_bot(bot)
    guild = _Guild(3)
    guild.voice_client = _BadExistingVC(guild)
    join_ctx = _Ctx(
        author=_Member(voice=SimpleNamespace(channel=_Channel(_GoodNewVC()))),
        guild=guild,
    )
    join_ctx.followup.send = _followup_send.__get__(join_ctx.followup, object)
    asyncio.run(bot.commands["join"](join_ctx))
    assert join_ctx.followup.messages == ["Connected"]

    class _BadLeaveVC:
        def __init__(self):
            self.recording = True

        def stop_recording(self):
            raise RuntimeError("stop failed")

        async def disconnect(self, force=True):
            raise RuntimeError("disconnect failed")

    discord_mod._voice_clients[9] = _BadLeaveVC()
    discord_mod._playback_tasks[9] = SimpleNamespace(cancel=lambda: None)
    leave_ctx = _Ctx(author=_Member(voice=None), guild=SimpleNamespace(id=9))
    asyncio.run(bot.commands["leave"](leave_ctx))
    assert leave_ctx.responses == ["Disconnected"]


def test_qwen_component_interrupt_branches(monkeypatch) -> None:
    import src.core.conduit.qwen_tts.component as comp_mod

    comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    comp._voice_paths = {"demo": Path("demo.pt")}
    audio_sent = []

    class _WorkerQueue:
        def __init__(self):
            self.items = [(0, "cancel")]

        def get(self, timeout=None):
            if self.items:
                return self.items.pop(0)
            comp.stop_event.set()
            raise comp_mod.Empty

        def empty(self):
            return True

    class _TopCancelEvent:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

        def set(self):
            self.calls = 99

    comp._tts = SimpleNamespace(
        sample_rate=24000,
        generate_streaming=lambda text, voice_path, language: iter(
            [
                (np.array([0.1], dtype=np.float32), {}),
                (np.array([0.2], dtype=np.float32), {}),
            ]
        ),
    )
    comp._task_queue = _WorkerQueue()
    monkeypatch.setattr(comp_mod.threading, "Event", _TopCancelEvent)
    comp._worker(
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda frame: audio_sent.append(frame)),
            text=SimpleNamespace(send=lambda frame: None),
        )
    )
    assert len(audio_sent) == 1

    comp2 = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    comp2._voice_paths = {"demo": Path("demo.pt")}
    mismatch_audio = []

    class _MismatchQueue:
        def __init__(self):
            self.items = [(0, "run")]

        def get(self, timeout=None):
            if self.items:
                return self.items.pop(0)
            comp2.stop_event.set()
            raise comp_mod.Empty

        def empty(self):
            return True

    def _mismatch_stream(*args, **kwargs):
        yield np.array([0.1], dtype=np.float32), {}
        comp2._generation = 1
        yield np.array([0.2], dtype=np.float32), {}

    comp2._tts = SimpleNamespace(sample_rate=24000, generate_streaming=_mismatch_stream)
    comp2._task_queue = _MismatchQueue()
    comp2._worker(
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda frame: mismatch_audio.append(frame)),
            text=SimpleNamespace(send=lambda frame: None),
        )
    )
    assert len(mismatch_audio) == 1

    run_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    run_comp._load_model = lambda: None
    run_comp._register_voices = lambda: None
    run_comp._worker = lambda outputs: None
    monkeypatch.setattr(
        comp_mod.threading,
        "Thread",
        lambda target, args=(), daemon=False, **kwargs: SimpleNamespace(
            start=lambda: target(*args),
            join=lambda timeout=None: None,
        ),
    )

    class _InterruptQueue:
        def __init__(self):
            self._empty = False

        def empty(self):
            return self._empty is False

        def get_nowait(self):
            self._empty = True
            raise comp_mod.Empty

        def put(self, item):
            return None

    run_comp._task_queue = _InterruptQueue()
    run_comp.run(
        comp_mod.QwenTTSInputs(
            text=_FakeRecv([EOS.END, None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="stop"), None]),
        ),
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda frame: None),
            text=SimpleNamespace(send=lambda frame: None),
        ),
    )


def test_object_segmenter_empty_keep_boxes() -> None:
    import src.core.conduit.yolo.object_segmenter as seg_mod

    segmenter = seg_mod.ObjectSegmenter(seg_mod.ObjectSegmenterConfig())
    segmenter._prompts = ["cat"]

    class _Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value)

        def int(self):
            return self

    class _Boxes:
        def __init__(self):
            self.xyxy = _Tensor([[0, 0, 1, 1]])
            self.conf = _Tensor([0.9])
            self.cls = _Tensor([5])
            self.id = None
            self.is_track = False

        def __len__(self):
            return 1

    segmenter._model = SimpleNamespace(
        predict=lambda frame, **kwargs: [SimpleNamespace(boxes=_Boxes(), masks=None)]
    )
    segmenter._tracker_yaml = None
    assert segmenter._infer(np.zeros((2, 2, 3), dtype=np.uint8)) is None


def test_pose_renderer_3d_neg_z_branch(monkeypatch) -> None:
    import src.core.conduit.pose_renderer_3d as pose3d_mod

    class _FakeMesh:
        def __init__(self, vertices=None, faces=None, process=False):
            self.vertices = (
                np.asarray(vertices) if vertices is not None else np.zeros((1, 3))
            )
            self.faces = (
                np.asarray(faces)
                if faces is not None
                else np.zeros((1, 3), dtype=np.int32)
            )
            self.transforms = []
            self.bounds = np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
            )

        def apply_transform(self, mat):
            self.transforms.append(mat)

    renderer = object.__new__(pose3d_mod.PoseRenderer3D)
    renderer.config = pose3d_mod.PoseRenderer3DConfig(
        width=4, height=4, camera_distance=2.0, device="cpu"
    )
    renderer._faces = np.array([[0, 1, 2]], dtype=np.int32)
    renderer._scene = SimpleNamespace(nodes=[])
    renderer._scene.add = (
        lambda obj, name=None, pose=None: renderer._scene.nodes.append(
            SimpleNamespace(obj=obj, name=name, pose=pose)
        )
        or renderer._scene.nodes[-1]
    )
    renderer._scene.remove_node = lambda node: renderer._scene.nodes.remove(node)
    renderer._renderer = SimpleNamespace(
        render=lambda scene, flags=None: (np.zeros((4, 4, 4), dtype=np.uint8), None)
    )
    renderer._pyrender = SimpleNamespace(
        MetallicRoughnessMaterial=lambda **kwargs: SimpleNamespace(**kwargs),
        Mesh=SimpleNamespace(
            from_trimesh=lambda mesh, material=None: SimpleNamespace(
                mesh=mesh, material=material
            )
        ),
        PerspectiveCamera=lambda **kwargs: SimpleNamespace(**kwargs),
        RenderFlags=SimpleNamespace(RGBA="rgba"),
    )
    renderer._trimesh = SimpleNamespace(
        Trimesh=_FakeMesh,
        creation=SimpleNamespace(cylinder=lambda **kwargs: _FakeMesh()),
        transformations=SimpleNamespace(
            rotation_matrix=lambda angle, axis: np.eye(4, dtype=np.float32)
        ),
    )
    real_allclose = pose3d_mod.np.allclose
    state = {"forced": False}

    def _branch_allclose(a, b, *args, **kwargs):
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        if (
            not state["forced"]
            and np.array_equal(arr_a, np.array([0.0, 0.0, 1.0]))
            and np.array_equal(arr_b, np.array([0.0, 0.0, 1.0]))
        ):
            state["forced"] = True
            return False
        if (
            state["forced"]
            and np.array_equal(arr_a, np.array([0.0, 0.0, 1.0]))
            and np.array_equal(arr_b, np.array([0.0, 0.0, -1.0]))
        ):
            return True
        return real_allclose(a, b, *args, **kwargs)

    monkeypatch.setattr(pose3d_mod.np, "allclose", _branch_allclose)
    image = renderer._render_mesh(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert image.shape == (4, 4, 3)


def test_qwen_small_branches(monkeypatch, tmp_path: Path) -> None:
    import transformers.utils.generic as transformers_generic

    if not hasattr(transformers_generic, "check_model_inputs"):
        monkeypatch.setattr(
            transformers_generic,
            "check_model_inputs",
            lambda *args, **kwargs: (lambda func: func),
            raising=False,
        )

    import src.core.conduit.qwen_tts.model as model_mod
    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts as cfg_mod
    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts_tokenizer_v2 as cfg_v2_mod
    import src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts_tokenizer_v2 as tokv2_mod
    import src.core.conduit.qwen_tts.tts_model.qwen3_tts_tokenizer as tok_mod

    monkeypatch.setattr(cfg_mod, "rope_config_validation", lambda cfg: None)
    monkeypatch.setattr(cfg_mod, "layer_type_validation", lambda layer_types: None)
    cfg = cfg_mod.Qwen3TTSConfig()
    assert cfg.talker_config is not None and cfg.speaker_encoder_config is not None
    cfg_v2 = cfg_v2_mod.Qwen3TTSTokenizerV2Config()
    assert cfg_v2.encoder_config is not None and cfg_v2.decoder_config is not None

    class _NoParamModel:
        def __init__(self):
            self.config = SimpleNamespace(model_type="qwen3_tts_tokenizer_25hz")
            self.dtype = torch.float32

        def parameters(self):
            if False:
                yield None
            return

        def decode(self, audio_codes, xvectors, ref_mels, return_dict=True):
            return SimpleNamespace(audio_values=[torch.ones(3, dtype=torch.float32)])

        def get_model_type(self):
            return self.config.model_type

        def get_output_sample_rate(self):
            return 24000

    tokenizer = object.__new__(tok_mod.Qwen3TTSTokenizer)
    tokenizer.model = _NoParamModel()
    tokenizer.config = tokenizer.model.config
    tokenizer.device = torch.device("cpu")
    tokenizer.feature_extractor = SimpleNamespace(sampling_rate=24000)

    monkeypatch.setattr(
        tok_mod, "urlparse", lambda _value: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert tokenizer._is_url("boom") is False
    wavs, _ = tok_mod.Qwen3TTSTokenizer.decode(
        tokenizer,
        {
            "audio_codes": [np.array([1, 2], dtype=np.int64)],
            "xvectors": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "ref_mels": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        },
    )
    assert len(wavs) == 1

    class _Processor:
        def __call__(self, text=None, return_tensors=None, padding=None):
            return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    class _SpeechTokenizer:
        def __init__(self):
            self.model = SimpleNamespace(
                decoder=lambda codes: torch.arange(len(codes), dtype=torch.float32)
            )

        def encode(self, wavs, sr=None):
            return SimpleNamespace(
                audio_codes=[
                    torch.tensor([1, 2], dtype=torch.long) for _ in wavs
                ]
            )

        def get_output_sample_rate(self):
            return 24000

    class _Model:
        def __init__(self):
            self.generate_config = {}
            self.tts_model_type = "base"
            self.speech_tokenizer = _SpeechTokenizer()
            self.speaker_encoder_sample_rate = 24000
            self.talker = SimpleNamespace()

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def extract_speaker_embedding(self, audio, sr):
            return torch.ones(4, dtype=torch.float32)

        def generate(self, **kwargs):
            return kwargs

    qwen = model_mod.Qwen3TTSModel(_Model(), _Processor())
    prompt_item = model_mod.VoiceClonePromptItem(
        ref_code=torch.tensor([1], dtype=torch.long),
        ref_spk_embedding=torch.tensor([0.1], dtype=torch.float32),
        ref_text="hello",
        x_vector_only_mode=False,
        icl_mode=True,
    )
    out = qwen.generate_voice_clone(
        ["a", "b"], language=["English"], voice_clone_prompt=[prompt_item]
    )
    assert out["languages"] == ["English", "English"]
    with pytest.raises(ValueError):
        qwen.generate_voice_clone(
            ["a", "b"],
            voice_clone_prompt=[prompt_item, prompt_item, prompt_item],
        )

    streaming = model_mod.SimpleStreamingTTS(_Model(), _Processor(), torch.device("cpu"))
    monkeypatch.setattr(model_mod.torch, "load", lambda path, weights_only=False: "prompt")

    class _Talker:
        def __init__(self):
            self.forward = self._forward
            self.calls = 0

        def _forward(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                hidden_states=(None, torch.tensor([self.calls], dtype=torch.float32))
            )

    streaming._talker = _Talker()
    streaming._model.generate_voice_clone = (
        lambda **kwargs: [streaming._talker.forward(), streaming._talker.forward()]
    )
    streaming._decode_audio = lambda codes: np.arange(len(codes), dtype=np.float32)
    monkeypatch.setattr(
        model_mod.threading,
        "Thread",
        lambda target, args=(), daemon=False, **kwargs: SimpleNamespace(
            start=lambda: target(*args)
        ),
    )
    monkeypatch.setattr(
        model_mod,
        "_streaming_config",
        model_mod.StreamingConfig(min_initial_frames=2, yield_every_n_frames=2),
    )
    outputs = list(streaming.generate_streaming("hi", tmp_path / "voice.pt"))
    assert outputs[-1][1]["is_final"] is True

    monkeypatch.setitem(
        tokv2_mod.ROPE_INIT_FUNCTIONS,
        "default",
        lambda config, device=None: (
            torch.ones(config.head_dim // 2, dtype=torch.float32),
            1.0,
        ),
    )
    no_proj = tokv2_mod.ResidualVectorQuantizer(
        dimension=2,
        input_dimension=2,
        output_dimension=2,
        n_q=2,
        bins=4,
        force_projection=False,
    )
    assert isinstance(no_proj.input_proj, torch.nn.Identity)
    assert isinstance(no_proj.output_proj, torch.nn.Identity)
    encoder = tokv2_mod.Qwen3TTSTokenizerV2Encoder(cfg_v2.encoder_config)
    assert encoder.decoder is None

    decoder_cfg = cfg_v2_mod.Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=8,
        hidden_size=4,
        latent_dim=4,
        max_position_embeddings=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        sliding_window=2,
        intermediate_size=8,
        hidden_act="relu",
        num_hidden_layers=1,
        num_quantizers=2,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
        decoder_dim=4,
        head_dim=4,
        codebook_dim=4,
    )
    decoder_cfg._attn_implementation = "eager"
    transformer = tokv2_mod.Qwen3TTSTokenizerV2DecoderTransformerModel(decoder_cfg)
    with pytest.raises(ValueError):
        transformer(input_ids=torch.ones((1, 2), dtype=torch.long))
    out = transformer(inputs_embeds=torch.ones((1, 2, 4)), use_cache=True)
    assert out.last_hidden_state.shape == (1, 2, 4)


def test_gaussian_diffusion_paths(monkeypatch) -> None:
    import sys
    import src.core.conduit.dart_control.diffusion.gaussian_diffusion as gd_mod

    assert gd_mod.get_named_beta_schedule("linear", 4).shape == (4,)
    assert gd_mod.get_named_beta_schedule("cosine", 4).shape == (4,)
    with pytest.raises(NotImplementedError):
        gd_mod.get_named_beta_schedule("bad", 4)
    assert gd_mod.betas_for_alpha_bar(4, lambda t: 1.0 - t * 0.1).shape == (4,)
    assert gd_mod.LossType.RESCALED_KL.is_vb() is True
    assert gd_mod.LossType.MSE.is_vb() is False

    diff = gd_mod.GaussianDiffusion(
        betas=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        model_mean_type=gd_mod.ModelMeanType.START_X,
        model_var_type=gd_mod.ModelVarType.FIXED_SMALL,
        loss_type=gd_mod.LossType.MSE,
        rescale_timesteps=True,
    )
    x_start = torch.ones((2, 1, 2, 2), dtype=torch.float32)
    t = torch.tensor([0, 1], dtype=torch.long)
    assert gd_mod._extract_into_tensor(np.array([1.0, 2.0]), torch.tensor([0]), (1, 1)).shape == (1, 1)
    assert diff.q_mean_variance(x_start, t)[0].shape == x_start.shape
    noisy = diff.q_sample(x_start, t, noise=torch.zeros_like(x_start))
    assert noisy.shape == x_start.shape
    assert diff.q_posterior_mean_variance(x_start, noisy, t)[0].shape == x_start.shape

    class _BaseModel(torch.nn.Module):
        def __init__(self, out_channels):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.out_channels = out_channels
            self.num_classes = 5

        def forward(self, x, ts, **kwargs):
            return torch.zeros((x.shape[0], self.out_channels, *x.shape[2:]), dtype=x.dtype, device=x.device)

    fixed_model = _BaseModel(1)
    assert diff.p_mean_variance(fixed_model, x_start, t, denoised_fn=lambda x: x + 2.0)["mean"].shape == x_start.shape
    assert diff.p_sample(fixed_model, x_start, t, const_noise=True)["sample"].shape == x_start.shape
    assert diff._scale_timesteps(t).dtype == torch.float32
    monkeypatch.setitem(sys.modules, "tqdm.auto", SimpleNamespace(tqdm=lambda x: x))
    loop_out = diff.p_sample_loop(
        fixed_model,
        x_start.shape,
        noise=torch.zeros_like(x_start),
        model_kwargs={"y": torch.zeros((2, 1), dtype=torch.long)},
        progress=True,
        dump_steps=[0],
        skip_timesteps=1,
        init_image=torch.zeros_like(x_start),
        randomize_class=True,
        const_noise=True,
    )
    assert isinstance(loop_out, list)
    assert list(
        diff.p_sample_loop_progressive(
            fixed_model,
            x_start.shape,
            noise=torch.zeros_like(x_start),
            model_kwargs={"y": torch.zeros((2, 1), dtype=torch.long)},
            progress=True,
            skip_timesteps=1,
            init_image=torch.zeros_like(x_start),
            randomize_class=True,
            const_noise=True,
        )
    )

    diff_eps = gd_mod.GaussianDiffusion(
        betas=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        model_mean_type=gd_mod.ModelMeanType.EPSILON,
        model_var_type=gd_mod.ModelVarType.LEARNED_RANGE,
        loss_type=gd_mod.LossType.MSE,
    )
    learned_model = _BaseModel(2)
    assert diff_eps.p_mean_variance(learned_model, x_start, t)["pred_xstart"].shape == x_start.shape
    assert diff_eps._predict_xstart_from_eps(x_start, t, torch.zeros_like(x_start)).shape == x_start.shape
    assert diff_eps._predict_eps_from_xstart(x_start, t, x_start).shape == x_start.shape
    assert diff_eps.ddim_sample(learned_model, x_start, t, eta=0.5)["sample"].shape == x_start.shape
    assert diff_eps.ddim_sample_loop(
        learned_model,
        x_start.shape,
        noise=torch.zeros_like(x_start),
        model_kwargs={"y": torch.zeros((2, 1), dtype=torch.long)},
        progress=True,
        eta=0.1,
        skip_timesteps=1,
        init_image=torch.zeros_like(x_start),
        randomize_class=True,
    ).shape == x_start.shape
    assert list(
        diff_eps.ddim_sample_loop_progressive(
            learned_model,
            x_start.shape,
            noise=torch.zeros_like(x_start),
            model_kwargs={"y": torch.zeros((2, 1), dtype=torch.long)},
            progress=True,
            eta=0.1,
            skip_timesteps=1,
            init_image=torch.zeros_like(x_start),
            randomize_class=True,
        )
    )

    diff_prev = gd_mod.GaussianDiffusion(
        betas=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        model_mean_type=gd_mod.ModelMeanType.PREVIOUS_X,
        model_var_type=gd_mod.ModelVarType.LEARNED,
        loss_type=gd_mod.LossType.MSE,
    )
    prev_model = _BaseModel(2)
    assert diff_prev.p_mean_variance(prev_model, x_start, t, clip_denoised=False)["mean"].shape == x_start.shape
    assert diff_prev._predict_xstart_from_xprev(x_start, t, x_start).shape == x_start.shape


def test_gaussian_diffusion_remaining_paths(monkeypatch) -> None:
    import src.core.conduit.dart_control.diffusion.gaussian_diffusion as gd_mod

    x_start = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    t = torch.tensor([0], dtype=torch.long)

    class _BaseModel(torch.nn.Module):
        def __init__(self, out_channels):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.out_channels = out_channels

        def forward(self, x, ts, **kwargs):
            return torch.zeros((x.shape[0], self.out_channels, *x.shape[2:]), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(gd_mod.th, "randn_like", lambda tensor: torch.zeros_like(tensor))
    monkeypatch.setattr(
        gd_mod.th,
        "randn",
        lambda *shape, device=None: torch.zeros(shape, dtype=torch.float32, device=device),
    )

    diff = gd_mod.GaussianDiffusion(
        betas=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        model_mean_type=gd_mod.ModelMeanType.START_X,
        model_var_type=gd_mod.ModelVarType.FIXED_SMALL,
        loss_type=gd_mod.LossType.MSE,
    )
    assert diff.q_sample(x_start, t).shape == x_start.shape
    assert diff.p_sample_loop(_BaseModel(1), x_start.shape, device=x_start.device).shape == x_start.shape
    assert list(
        diff.p_sample_loop_progressive(
            _BaseModel(1),
            x_start.shape,
            device=x_start.device,
            skip_timesteps=1,
        )
    )

    diff_eps = gd_mod.GaussianDiffusion(
        betas=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        model_mean_type=gd_mod.ModelMeanType.EPSILON,
        model_var_type=gd_mod.ModelVarType.LEARNED_RANGE,
        loss_type=gd_mod.LossType.MSE,
    )
    assert list(
        diff_eps.ddim_sample_loop_progressive(
            _BaseModel(2),
            x_start.shape,
            device=x_start.device,
            skip_timesteps=1,
        )
    )

    bad = gd_mod.GaussianDiffusion(
        betas=np.array([0.1, 0.2], dtype=np.float64),
        model_mean_type="bad",
        model_var_type=gd_mod.ModelVarType.FIXED_SMALL,
        loss_type=gd_mod.LossType.MSE,
    )
    with pytest.raises(NotImplementedError):
        bad.p_mean_variance(_BaseModel(1), x_start[:, :, :1, :1], torch.tensor([0], dtype=torch.long))


def test_smpl_utils_and_dart_control_branches(monkeypatch, tmp_path: Path) -> None:
    import pickle
    import src.core.conduit.dart_control.component as dart_comp_mod
    import src.core.conduit.dart_control.diffusion.nn as nn_mod
    import src.core.conduit.dart_control.rotation_conversions as rot_mod
    import src.core.conduit.dart_control.smpl_utils as smpl_mod

    joints_template = torch.zeros((1, 22, 3), dtype=torch.float32)
    joints_template[:, 1, :] = torch.tensor([0.0, 0.0, 0.0])
    joints_template[:, 2, :] = torch.tensor([1.0, 0.0, 0.0])

    class _FakeBodyModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(
            self,
            betas=None,
            global_orient=None,
            body_pose=None,
            transl=None,
            return_vertices=False,
            **kwargs,
        ):
            batch = transl.shape[0] if transl is not None else betas.shape[0]
            base = joints_template.repeat(batch, 1, 1)
            if transl is not None:
                transl_base = transl[:, 0] if transl.ndim == 3 else transl
                base = base + transl_base.unsqueeze(1)
            return SimpleNamespace(
                joints=base,
                vertices=torch.zeros((batch, 5, 3), dtype=torch.float32),
            )

    monkeypatch.setattr(smpl_mod.smplx, "build_layer", lambda *args, **kwargs: _FakeBodyModel())
    primitive = smpl_mod.PrimitiveUtility(device="cpu")
    assert primitive.get_smpl_model("male") is primitive.bm_male
    assert primitive.get_smpl_model("female") is primitive.bm_female

    poses_6d = rot_mod.matrix_to_rotation_6d(
        torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 3, 22, 1, 1)
    ).reshape(1, 3, 132)
    feature_dict = {
        "gender": "male",
        "betas": torch.zeros((1, 3, 10), dtype=torch.float32),
        "transf_rotmat": torch.eye(3).unsqueeze(0),
        "transf_transl": torch.zeros((1, 1, 3), dtype=torch.float32),
        "transl": torch.zeros((1, 3, 3), dtype=torch.float32),
        "transl_delta": torch.zeros((1, 3, 3), dtype=torch.float32),
        "joints": torch.zeros((1, 3, 66), dtype=torch.float32),
        "joints_delta": torch.zeros((1, 3, 66), dtype=torch.float32),
        "global_orient_delta_6d": rot_mod.matrix_to_rotation_6d(
            torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
        ),
        "poses_6d": poses_6d,
        "pelvis_delta": torch.zeros((1, 3), dtype=torch.float32),
    }
    assert smpl_mod.tensor_dict_to_device({"x": torch.ones(1), "y": "z"}, device="cpu")["x"].device.type == "cpu"
    aa_dict = smpl_mod.convert_smpl_aa_to_rotmat(
        {
            "global_orient": torch.zeros((1, 3), dtype=torch.float32),
            "body_pose": torch.zeros((1, 63), dtype=torch.float32),
        }
    )
    assert aa_dict["global_orient"].shape == (1, 3, 3)
    sixd_param = smpl_mod.get_smplx_param_from_6d(
        {
            "gender": "male",
            "transl": torch.zeros((2, 3), dtype=torch.float32),
            "betas": torch.zeros(10, dtype=torch.float32),
            "poses_6d": poses_6d[0, :2],
        }
    )
    assert sixd_param["body_pose"].shape == (2, 21, 3, 3)
    assert smpl_mod.get_new_coordinate(joints_template)[0].shape == (1, 3, 3)
    assert smpl_mod.update_global_transform(
        {
            "transf_rotmat": torch.eye(3).unsqueeze(0),
            "transf_transl": torch.zeros((1, 1, 3), dtype=torch.float32),
        },
        torch.eye(3).unsqueeze(0),
        torch.zeros((1, 1, 3), dtype=torch.float32),
    )["transf_rotmat"].shape == (1, 3, 3)
    points = torch.ones((1, 2, 3), dtype=torch.float32)
    assert smpl_mod.transform_local_points_to_global(
        points, torch.eye(3).unsqueeze(0), torch.zeros((1, 1, 3))
    ).shape == points.shape
    assert smpl_mod.transform_global_points_to_local(
        points, torch.eye(3).unsqueeze(0), torch.zeros((1, 1, 3))
    ).shape == points.shape
    assert smpl_mod.get_dict_subset_by_batch(feature_dict, 0)["gender"] == "male"

    body_param = primitive.feature_dict_to_smpl_dict(feature_dict)
    assert body_param["global_orient"].shape == (1, 3, 3, 3)
    features_for_tensor = primitive.calc_features(body_param, use_predicted_joints=True)
    for key in ("transl_delta", "joints_delta", "global_orient_delta_6d"):
        features_for_tensor[key] = torch.cat(
            [features_for_tensor[key], features_for_tensor[key][:, -1:, :]], dim=1
        )
    assert primitive.dict_to_tensor(features_for_tensor).shape[-1] == primitive.feature_dim
    assert set(
        primitive.tensor_to_dict(
            torch.zeros((1, 1, primitive.feature_dim), dtype=torch.float32)
        )
    ) == set(primitive.motion_repr)
    assert primitive.smpl_dict_to_vertices(body_param).shape[:2] == (1, 3)
    assert primitive.smpl_dict_inference(
        {
            "gender": "male",
            "betas": torch.zeros((3, 10), dtype=torch.float32),
            "global_orient": torch.eye(3).reshape(1, 3, 3).repeat(3, 1, 1),
            "body_pose": torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 21, 1, 1),
            "transl": torch.zeros((3, 3), dtype=torch.float32),
        },
        return_vertices=False,
        batch_size=2,
    ).shape == (3, 22, 3)
    assert primitive.smpl_dict_inference(
        {
            "gender": "male",
            "betas": torch.zeros((3, 10), dtype=torch.float32),
            "global_orient": torch.eye(3).reshape(1, 3, 3).repeat(3, 1, 1),
            "body_pose": torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 21, 1, 1),
            "transl": torch.zeros((3, 3), dtype=torch.float32),
        },
        return_vertices=True,
        batch_size=2,
    )[1].shape[0] == 3

    monkeypatch.setattr(
        smpl_mod,
        "get_new_coordinate",
        lambda joints: (torch.eye(3).unsqueeze(0), torch.zeros((1, 1, 3))),
    )
    assert primitive.get_new_coordinate(body_param, use_predicted_joints=False)[0].shape == (1, 3, 3)
    assert primitive.get_new_coordinate(
        body_param, use_predicted_joints=True, pred_joints=joints_template
    )[0].shape == (1, 3, 3)
    assert primitive.calc_calibrate_offset(
        {"gender": "male", "betas": torch.zeros((1, 10), dtype=torch.float32)}
    ).shape == (1, 3)
    canonical = primitive.canonicalize(dict(body_param), use_predicted_joints=False)[2]
    assert primitive.calc_features(canonical, use_predicted_joints=True)["poses_6d"].shape[-1] == 132
    assert primitive.calc_features(canonical, use_predicted_joints=False)["joints"].shape[-1] == 66
    blended_pred = primitive.get_blended_feature(feature_dict, use_predicted_joints=True)
    blended_fk = primitive.get_blended_feature(feature_dict, use_predicted_joints=False)
    assert blended_pred[1]["transf_rotmat"].shape == (1, 3, 3)
    assert blended_fk[1]["transf_transl"].shape == (1, 1, 3)
    assert primitive.transform_feature_to_world(blended_pred[1])["transf_rotmat"].shape == (1, 3, 3)
    assert primitive.transform_primitive_to_world(dict(body_param))["transf_transl"].shape == (1, 1, 3)
    assert primitive.transform_primitive_to_world(
        {**dict(body_param), "joints": torch.zeros((1, 3, 22, 3), dtype=torch.float32)}
    )["transf_rotmat"].shape == (1, 3, 3)

    monkeypatch.setattr(nn_mod, "_AMP_DEVICE_TYPE", None)
    monkeypatch.setattr(nn_mod, "_custom_fwd", lambda fn: fn)
    monkeypatch.setattr(nn_mod, "_custom_bwd", lambda fn: fn)
    x = torch.tensor([2.0], requires_grad=True)
    nn_mod.checkpoint(lambda t: t * t, [x], [], True).sum().backward()
    assert x.grad is not None
    no_grad_ctx = SimpleNamespace(
        saved_tensors=[torch.tensor([1.0])],
        input_length=1,
        run_function=lambda v: v,
    )
    assert nn_mod.CheckpointFunction._backward_impl(no_grad_ctx, torch.tensor([1.0]))[2] is None
    no_output_ctx = SimpleNamespace(
        saved_tensors=[torch.tensor([1.0], requires_grad=True)],
        input_length=1,
        run_function=lambda v: v.detach(),
    )
    assert nn_mod.CheckpointFunction._backward_impl(no_output_ctx, torch.tensor([1.0]))[2] is None

    with pytest.raises(ValueError):
        rot_mod.euler_angles_to_matrix(torch.zeros((1, 3)), "XY")
    with pytest.raises(ValueError):
        rot_mod.matrix_to_euler_angles(torch.eye(3).unsqueeze(0), "XY")
    with pytest.raises(ValueError):
        rot_mod.matrix_to_euler_angles(torch.eye(3).unsqueeze(0), "XQZ")

    stand_path = tmp_path / "stand.pkl"
    with stand_path.open("wb") as fh:
        pickle.dump(
            {
                "transl": np.zeros((1, 3), dtype=np.float32),
                "global_orient": np.zeros((1, 3), dtype=np.float32),
                "body_pose": np.zeros((1, 63), dtype=np.float32),
            },
            fh,
        )

    dart = dart_comp_mod.DartControl(
        dart_comp_mod.DartControlConfig(
            device="cpu",
            batch_size=1,
            stand_path=str(stand_path),
            policy_checkpoint="",
        )
    )
    fake_engine = SimpleNamespace(
        history_shape=(2,),
        normalize=lambda tensor: tensor,
        encode_text=lambda items: torch.ones((1, 4)),
    )
    fake_putil = SimpleNamespace(
        calc_calibrate_offset=lambda data: torch.zeros((1, 3), dtype=torch.float32),
        canonicalize=lambda primitive_dict: (
            torch.eye(3).unsqueeze(0),
            torch.zeros((1, 1, 3)),
            primitive_dict,
        ),
        calc_features=lambda primitive_dict: {
            "transl": primitive_dict["transl"],
            "transl_delta": torch.zeros((1, 2, 3), dtype=torch.float32),
            "joints": torch.zeros((1, 3, 66), dtype=torch.float32),
            "joints_delta": torch.zeros((1, 2, 66), dtype=torch.float32),
            "global_orient_delta_6d": torch.zeros((1, 2, 6), dtype=torch.float32),
            "poses_6d": torch.zeros((1, 3, 132), dtype=torch.float32),
        },
        dict_to_tensor=lambda feature_dict: torch.zeros((1, 3, 276), dtype=torch.float32),
    )
    dart._init_from_stand(fake_engine, fake_putil)
    assert dart._history.shape == (1, 2, 276)
    assert dart._load_policy(SimpleNamespace(history_shape=(2,), noise_shape=(1, 3))) is None
    missing = dart_comp_mod.DartControl(
        dart_comp_mod.DartControlConfig(
            device="cpu",
            batch_size=1,
            policy_checkpoint=str(tmp_path / "missing.pt"),
        )
    )
    assert missing._load_policy(SimpleNamespace(history_shape=(2,), noise_shape=(1, 3))) is None

    cancel_dart = dart_comp_mod.DartControl(
        dart_comp_mod.DartControlConfig(device="cpu", batch_size=1, fps=60)
    )
    cancel_dart._stop_event = _OneShotStopEvent()
    cancel_dart._ensure_engine = lambda: fake_engine
    cancel_dart._ensure_primitive_util = lambda: SimpleNamespace()
    cancel_dart._load_policy = lambda engine: None
    cancel_dart._init_from_stand = lambda engine, putil: None
    futures = []

    class _Executor:
        def __init__(self, *args, **kwargs):
            return None

        def submit(self, fn, *args):
            future = _FakeFuture(torch.zeros((1, 1, 276), dtype=torch.float32))
            futures.append(future)
            return future

        def shutdown(self, wait=False):
            return None

    monkeypatch.setattr(dart_comp_mod, "ThreadPoolExecutor", _Executor)
    calls = {"generate": 0}

    def _generate(engine, putil, text_embedding, policy, current_goal):
        calls["generate"] += 1
        if calls["generate"] >= 2:
            cancel_dart.stop_event.set()
        return torch.zeros((1, 1, 276), dtype=torch.float32)

    cancel_dart._generate_primitive = _generate
    cancel_dart._prepare_frames = lambda world_features: [{"waist": None}]
    sent = []
    cancel_dart.run(
        dart_comp_mod.DartControlInputs(
            goal=_FakeRecv([GoalFrame.new(x=1.0, y=2.0, z=3.0), None]),
            instruction=_FakeRecv(
                [TextFrame.new(text="walk"), TextFrame.new(text="run"), None]
            ),
        ),
        dart_comp_mod.DartControlOutputs(
            motion=SimpleNamespace(send=lambda frame: sent.append(frame))
        ),
    )
    assert sent and any(f.cancelled for f in futures)

    error_dart = dart_comp_mod.DartControl(
        dart_comp_mod.DartControlConfig(device="cpu", batch_size=1)
    )
    error_dart._stop_event = _OneShotStopEvent()
    error_dart._ensure_engine = lambda: fake_engine
    error_dart._ensure_primitive_util = lambda: SimpleNamespace()
    error_dart._load_policy = lambda engine: None
    error_dart._init_from_stand = lambda engine, putil: None
    monkeypatch.setattr(dart_comp_mod, "ThreadPoolExecutor", _Executor)

    def _boom(engine, putil, text_embedding, policy, current_goal):
        error_dart.stop_event.set()
        raise RuntimeError("boom")

    error_dart._generate_primitive = _boom
    error_dart._prepare_frames = lambda world_features: []
    error_dart.run(
        dart_comp_mod.DartControlInputs(
            goal=_FakeRecv([None]),
            instruction=_FakeRecv([TextFrame.new(text="walk"), None]),
        ),
        dart_comp_mod.DartControlOutputs(
            motion=SimpleNamespace(send=lambda frame: None)
        ),
    )


def test_smpl_utils_matrix_betas_branch() -> None:
    import src.core.conduit.dart_control.rotation_conversions as rot_mod
    import src.core.conduit.dart_control.smpl_utils as smpl_mod

    poses_6d = rot_mod.matrix_to_rotation_6d(
        torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 2, 22, 1, 1)
    ).reshape(2, 132)
    params = smpl_mod.get_smplx_param_from_6d(
        {
            "gender": "male",
            "transl": torch.zeros((2, 3), dtype=torch.float32),
            "betas": torch.zeros((2, 12), dtype=torch.float32),
            "poses_6d": poses_6d,
        }
    )
    assert params["betas"].shape == (2, 10)
