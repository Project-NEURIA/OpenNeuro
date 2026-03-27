from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from pathlib import Path
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.frames import (
    AudioFrame,
    BodyPoseFrame,
    BonePose,
    CameraParamsFrame,
    InterruptFrame,
    ObjectSegmentationFrame,
    StereoCameraParamsFrame,
    StereoVideoFrame,
    TextFrame,
    VideoDataFormat,
    VideoFrame,
)


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


class _SyncThread:
    def __init__(self, target, args=(), daemon=False, **kwargs):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.kwargs = kwargs

    def start(self):
        self.target(*self.args)

    def join(self, timeout=None):
        return None


def test_monocular_depth_estimator_paths(monkeypatch) -> None:
    import torch
    import src.core.conduit.monocular_depth_estimator as mono_mod

    monkeypatch.setattr(mono_mod, "auto_device", lambda device: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(mono_mod, "auto_dtype", lambda device: "float32")
    monkeypatch.setattr(
        mono_mod,
        "resize_and_crop",
        lambda arr, width, height: np.asarray(arr)[:height, :width].copy(),
    )

    depth_pkg = types.ModuleType("depth_anything_3")
    api_mod = types.ModuleType("depth_anything_3.api")
    utils_pkg = types.ModuleType("depth_anything_3.utils")
    align_mod = types.ModuleType("depth_anything_3.utils.alignment")

    class _DepthModel:
        def __init__(self):
            self.to_device = None
            self.eval_called = False
            self.pred = SimpleNamespace(
                depth=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
                is_metric=True,
            )

        def to(self, device=None):
            self.to_device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def inference(self, **kwargs):
            self.kwargs = kwargs
            return self.pred

    fake_model = _DepthModel()
    api_mod.DepthAnything3 = type(
        "DepthAnything3",
        (),
        {"from_pretrained": classmethod(lambda cls, model_id: fake_model)},
    )
    align_mod.apply_metric_scaling = lambda depth_t, K_t: depth_t + 1.5
    depth_pkg.api = api_mod
    utils_pkg.alignment = align_mod
    monkeypatch.setitem(sys.modules, "depth_anything_3", depth_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api_mod)
    monkeypatch.setitem(sys.modules, "depth_anything_3.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.utils.alignment", align_mod)

    estimator = mono_mod.MonocularDepthEstimator(
        mono_mod.MonocularDepthEstimatorConfig(
            process_res=8,
            output_width=2,
            output_height=2,
        )
    )
    assert mono_mod.MonocularDepthEstimator.get_options({})["config"]["model"][0]["value"]

    estimator.setup()
    assert fake_model.to_device.type == "cpu"
    assert fake_model.eval_called is True

    metric = estimator._infer(np.zeros((2, 2, 3), dtype=np.uint8), np.eye(3, dtype=np.float32))
    assert metric.shape == (2, 2)

    fake_model.pred = SimpleNamespace(
        depth=np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32),
        is_metric=False,
    )
    scaled = estimator._infer(np.zeros((2, 2, 3), dtype=np.uint8), np.eye(3, dtype=np.float32))
    assert np.allclose(scaled, np.array([[3.5, 5.5], [7.5, 9.5]], dtype=np.float32))

    video = VideoFrame.new(data=np.zeros((2, 2, 3), dtype=np.uint8), format=VideoDataFormat.BGR)
    depth_out, video_out = estimator._resize_outputs(np.ones((2, 2), dtype=np.float32), video)
    assert depth_out.shape == (2, 2)
    assert video_out.shape == (2, 2, 3)

    sent_depth = []
    sent_video = []
    monkeypatch.setattr(estimator, "_infer", lambda rgb, intrinsics: np.ones((2, 2), dtype=np.float32))
    estimator.run(
        mono_mod.MonocularDepthEstimatorInputs(
            video=_FakeRecv([video, None]),
            camera_params=_FakeRecv(
                [
                    CameraParamsFrame.new(
                        intrinsics=np.eye(3, dtype=np.float32),
                        extrinsics=np.eye(4, dtype=np.float32),
                        width=2,
                        height=2,
                    )
                ]
            ),
        ),
        mono_mod.MonocularDepthEstimatorOutputs(
            depth=SimpleNamespace(send=lambda value: sent_depth.append(value)),
            video=SimpleNamespace(send=lambda value: sent_video.append(value)),
        ),
    )
    assert len(sent_depth) == 1
    assert len(sent_video) == 1

    estimator.run(
        mono_mod.MonocularDepthEstimatorInputs(
            video=_FakeRecv([video]),
            camera_params=_FakeRecv([None]),
        ),
        mono_mod.MonocularDepthEstimatorOutputs(
            depth=SimpleNamespace(send=lambda value: sent_depth.append(value)),
            video=SimpleNamespace(send=lambda value: sent_video.append(value)),
        ),
    )


def test_pose_renderer_paths() -> None:
    import src.core.conduit.pose_renderer as pose_mod

    assert pose_mod._project(BonePose(pos_x=1.0, pos_y=1.0), 100, 80) == (200, -90)

    poses = {
        "waist": BonePose(pos_x=0.0, pos_y=0.0),
        "chest": BonePose(pos_x=0.0, pos_y=0.5),
        "head": BonePose(pos_x=0.0, pos_y=1.0),
        "left_shoulder": BonePose(pos_x=-0.2, pos_y=0.5),
        "left_elbow": BonePose(pos_x=-0.4, pos_y=0.45),
        "left_hand": BonePose(pos_x=-0.6, pos_y=0.4),
        "right_shoulder": BonePose(pos_x=0.2, pos_y=0.5),
        "right_elbow": BonePose(pos_x=0.4, pos_y=0.45),
        "right_hand": BonePose(pos_x=0.6, pos_y=0.4),
        "left_knee": BonePose(pos_x=-0.15, pos_y=-0.5),
        "left_foot": BonePose(pos_x=-0.15, pos_y=-1.0),
        "right_knee": BonePose(pos_x=0.15, pos_y=-0.5),
        "right_foot": BonePose(pos_x=0.15, pos_y=-1.0),
        "unknown": BonePose(pos_x=0.1, pos_y=0.1),
        "missing": None,
    }

    renderer = pose_mod.PoseRenderer(pose_mod.PoseRendererConfig(width=160, height=120))
    img = renderer._render_pose(poses)
    assert img.shape == (120, 160, 3)
    assert img.dtype == np.uint8
    assert not np.all(img == np.array(pose_mod._BG_COLOR, dtype=np.uint8))

    sent = []
    renderer.run(
        pose_mod.PoseRendererInputs(
            pose=_FakeRecv([BodyPoseFrame(poses=poses), None])
        ),
        pose_mod.PoseRendererOutputs(
            video=SimpleNamespace(send=lambda value: sent.append(value))
        ),
    )
    assert len(sent) == 1
    assert sent[0].get(VideoDataFormat.BGR).shape == (120, 160, 3)


def test_stereo_to_mono_paths() -> None:
    import src.core.conduit.ffs_stereo_depth.stereo_to_mono as mono_mod

    left = np.zeros((2, 2, 3), dtype=np.uint8)
    right = np.ones((2, 2, 3), dtype=np.uint8)
    stereo = StereoVideoFrame.new(left=left, right=right, format=VideoDataFormat.RGB)
    assert mono_mod.StereoToMonocularVideo.get_options({})["config"]["eye"][1]["value"] == "right"

    sent_left = []
    mono_mod.StereoToMonocularVideo(
        mono_mod.StereoToMonocularVideoConfig(eye="left")
    ).run(
        mono_mod.StereoToMonocularVideoInputs(stereo_video=_FakeRecv([stereo, None])),
        mono_mod.StereoToMonocularVideoOutputs(
            video=SimpleNamespace(send=lambda value: sent_left.append(value))
        ),
    )
    assert np.array_equal(sent_left[0].get(VideoDataFormat.RGB), left)

    sent_right = []
    mono_mod.StereoToMonocularVideo(
        mono_mod.StereoToMonocularVideoConfig(eye="right")
    ).run(
        mono_mod.StereoToMonocularVideoInputs(stereo_video=_FakeRecv([stereo, None])),
        mono_mod.StereoToMonocularVideoOutputs(
            video=SimpleNamespace(send=lambda value: sent_right.append(value))
        ),
    )
    assert np.array_equal(sent_right[0].get(VideoDataFormat.RGB), right)


def test_stereo_depth_estimator_paths(monkeypatch) -> None:
    import cv2
    import src.core.conduit.ffs_stereo_depth.stereo_depth_estimator as stereo_mod

    monkeypatch.setattr(stereo_mod, "auto_device", lambda device: "cpu")
    monkeypatch.setattr(
        stereo_mod,
        "resize_and_crop",
        lambda arr, width, height: np.asarray(arr)[:height, :width].copy(),
    )

    class _FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def float(self):
            return self

        def permute(self, *dims):
            return _FakeTensor(self.value.transpose(dims))

        def unsqueeze(self, axis):
            return _FakeTensor(np.expand_dims(self.value, axis))

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value)

        @property
        def data(self):
            return self

    class _FakeBuffer(_FakeTensor):
        def copy_(self, other):
            self.value = np.asarray(other.value, dtype=np.float32)
            return self

        @property
        def shape(self):
            return self.value.shape

    class _FakePadder:
        def __init__(self, shape, divis_by, force_square):
            self.shape = shape
            self.divis_by = divis_by
            self.force_square = force_square

        def pad(self, t0, t1):
            return t0, t1

        def unpad(self, disp):
            return disp

    class _FakeModel:
        def __init__(self):
            self.args = SimpleNamespace(valid_iters=None, max_disp=None)
            self.calls = []

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def forward(self, t0, t1, **kwargs):
            self.calls.append((t0.value.shape, t1.value.shape, kwargs))
            h, w = t0.value.shape[-2:]
            return _FakeTensor(np.arange(h * w, dtype=np.float32).reshape(1, 1, h, w) - 1)

    fake_model = _FakeModel()
    fake_torch = types.SimpleNamespace(
        empty=lambda *shape, dtype=None, device=None: _FakeBuffer(np.zeros(shape, dtype=np.float32)),
        as_tensor=lambda value, device=None: _FakeTensor(value),
        no_grad=lambda: contextlib.nullcontext(),
        amp=SimpleNamespace(autocast=lambda *args, **kwargs: contextlib.nullcontext()),
        float16="float16",
        float32="float32",
        load=lambda path, map_location=None, weights_only=None: fake_model,
        set_grad_enabled=lambda enabled: None,
        _dynamo=SimpleNamespace(config=SimpleNamespace(disable=False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    core_pkg = types.ModuleType("core")
    core_utils_pkg = types.ModuleType("core.utils")
    core_utils_mod = types.ModuleType("core.utils.utils")
    core_utils_mod.InputPadder = _FakePadder
    monkeypatch.setitem(sys.modules, "core", core_pkg)
    monkeypatch.setitem(sys.modules, "core.utils", core_utils_pkg)
    monkeypatch.setitem(sys.modules, "core.utils.utils", core_utils_mod)

    estimator = stereo_mod.StereoDepthEstimator(
        stereo_mod.StereoDepthEstimatorConfig(
            resize_width=2,
            resize_height=2,
            valid_iters=3,
            max_disp=10,
        )
    )
    estimator.setup()
    assert fake_model.args.valid_iters == 3
    assert fake_model.args.max_disp == 10
    assert fake_model.device == "cpu"

    estimator._ensure_buffers(2, 2)
    first_buf = estimator._t0_buf
    estimator._ensure_buffers(2, 2)
    assert estimator._t0_buf is first_buf
    estimator._ensure_buffers(3, 2)
    assert estimator._buf_h == 3

    resized = stereo_mod.StereoDepthEstimator._resize_for_inference(
        np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3),
        2,
        2,
    )
    assert resized.shape == (2, 2, 3)

    disp = estimator._infer(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
    )
    assert disp.shape == (2, 2)
    assert disp.min() >= 0

    stereo_frame = StereoVideoFrame.new(
        left=np.zeros((2, 2, 3), dtype=np.uint8),
        right=np.ones((2, 2, 3), dtype=np.uint8) * 255,
        format=VideoDataFormat.RGB,
    )
    left_out, right_out = estimator._resize_outputs(stereo_frame)
    assert np.array_equal(left_out, cv2.cvtColor(stereo_frame.left, cv2.COLOR_RGB2BGR))
    assert np.array_equal(right_out, cv2.cvtColor(stereo_frame.right, cv2.COLOR_RGB2BGR))

    sent_depth = []
    sent_video = []
    monkeypatch.setattr(estimator, "_infer", lambda left, right: np.full((2, 2), 2.0, dtype=np.float32))
    monkeypatch.setattr(
        estimator,
        "_resize_outputs",
        lambda frame: (frame.left.copy(), frame.right.copy()),
    )
    camera = StereoCameraParamsFrame.new(
        intrinsics=np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
        baseline=0.5,
        width=2,
        height=2,
    )
    estimator.run(
        stereo_mod.StereoDepthEstimatorInputs(
            stereo_video=_FakeRecv([stereo_frame, None]),
            camera_params=_FakeRecv([camera]),
        ),
        stereo_mod.StereoDepthEstimatorOutputs(
            depth=SimpleNamespace(send=lambda value: sent_depth.append(value)),
            stereo_video=SimpleNamespace(send=lambda value: sent_video.append(value)),
        ),
    )
    assert len(sent_depth) == 1
    assert len(sent_video) == 1
    assert np.allclose(sent_depth[0].data, np.ones((2, 2), dtype=np.float32))

    estimator.run(
        stereo_mod.StereoDepthEstimatorInputs(
            stereo_video=_FakeRecv([stereo_frame]),
            camera_params=_FakeRecv([None]),
        ),
        stereo_mod.StereoDepthEstimatorOutputs(
            depth=SimpleNamespace(send=lambda value: sent_depth.append(value)),
            stereo_video=SimpleNamespace(send=lambda value: sent_video.append(value)),
        ),
    )


def test_object_segmenter_paths(monkeypatch, tmp_path: Path) -> None:
    import src.core.conduit.yolo.object_segmenter as seg_mod

    monkeypatch.setattr(seg_mod, "auto_device", lambda device: "cpu")
    monkeypatch.setattr(
        seg_mod,
        "resize_and_crop",
        lambda arr, width, height: np.asarray(arr)[:height, :width].copy(),
    )
    monkeypatch.setattr(seg_mod.os, "getcwd", lambda: str(tmp_path))
    chdir_calls = []
    monkeypatch.setattr(seg_mod.os, "chdir", lambda path: chdir_calls.append(str(path)))

    class _Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value)

        def int(self):
            return self

        def __len__(self):
            return len(self.value)

    class _Boxes:
        def __init__(self, xyxy, conf, cls, ids=None, is_track=False):
            self.xyxy = _Tensor(xyxy)
            self.conf = _Tensor(conf)
            self.cls = _Tensor(cls)
            self.id = _Tensor(ids) if ids is not None else None
            self.is_track = is_track

        def __len__(self):
            return len(self.cls.value)

    class _Masks:
        def __init__(self, data):
            self.data = _Tensor(data)

    class _Result:
        def __init__(self, boxes, masks):
            self.boxes = boxes
            self.masks = masks

    class _YOLOE:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
            self.class_sets = []
            self.responses = []

        def set_classes(self, prompts):
            self.class_sets.append(list(prompts))

        def track(self, frame, persist=True, tracker=None, **kwargs):
            self.last_track = (frame.shape, persist, tracker, kwargs)
            return [self.responses.pop(0)]

        def predict(self, frame, **kwargs):
            self.last_predict = (frame.shape, kwargs)
            return [self.responses.pop(0)]

    fake_yolo = _YOLOE("unused")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=lambda checkpoint: fake_yolo))

    segmenter = seg_mod.ObjectSegmenter(
        seg_mod.ObjectSegmenterConfig(
            model="small",
            conf=0.2,
            new_track_thresh=0.3,
            track_high_thresh=0.4,
            track_low_thresh=0.1,
            match_thresh=0.9,
        )
    )
    yaml_path = Path(segmenter._build_tracker_yaml())
    assert yaml_path.is_file()
    assert "track_high_thresh: 0.4" in yaml_path.read_text(encoding="utf-8")

    segmenter.setup()
    assert segmenter._model is fake_yolo
    assert segmenter._tracker_yaml is not None

    segmenter._set_phrases([" cat ", "dog", "cat", ""])
    assert segmenter._prompts == ["cat", "dog"]
    segmenter._set_phrases(["cat", "dog"])
    assert fake_yolo.class_sets == [["cat", "dog"]]
    assert len(chdir_calls) == 2

    fake_yolo.responses = [_Result(None, None)]
    assert segmenter._infer(np.zeros((2, 2, 3), dtype=np.uint8)) is None

    fake_yolo.responses = [
        _Result(
            _Boxes(
                xyxy=[[0, 0, 1, 1], [1, 1, 2, 2]],
                conf=[0.9, 0.8],
                cls=[0, 4],
                ids=[7, 8],
                is_track=True,
            ),
            _Masks(np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]], dtype=np.float32)),
        )
    ]
    masks, boxes, scores, object_ids, labels = segmenter._infer(
        np.zeros((2, 2, 3), dtype=np.uint8)
    )
    assert masks.shape == (1, 2, 2)
    assert boxes.shape == (1, 4)
    assert np.allclose(scores, np.array([0.9], dtype=np.float32))
    assert object_ids.tolist() == [7]
    assert labels == ("cat",)

    segmenter._tracker_yaml = None
    fake_yolo.responses = [
        _Result(
            _Boxes(
                xyxy=[[0, 0, 1, 1]],
                conf=[0.5],
                cls=[1],
                ids=None,
                is_track=False,
            ),
            None,
        )
    ]
    masks2, boxes2, scores2, ids2, labels2 = segmenter._infer(
        np.zeros((2, 2, 3), dtype=np.uint8)
    )
    assert masks2.shape == (1, 0, 0)
    assert boxes2.shape == (1, 4)
    assert scores2.tolist() == [0.5]
    assert ids2.tolist() == [0]
    assert labels2 == ("dog",)

    listener = seg_mod.ObjectSegmenter(seg_mod.ObjectSegmenterConfig())
    updated = []
    monkeypatch.setattr(listener, "_set_phrases", lambda prompts: updated.append(list(prompts)))
    listener._prompt_listener(
        seg_mod.ObjectSegmenterInputs(
            video=_FakeRecv([]),
            prompts=_FakeRecv([TextFrame.new(text="cat, dog"), None]),
        )
    )
    assert updated == [["cat", "dog"]]

    run_segmenter = seg_mod.ObjectSegmenter(seg_mod.ObjectSegmenterConfig())
    run_segmenter._lock = contextlib.nullcontext()
    run_segmenter._prompts = []
    run_segmenter._infer = lambda frame: None
    monkeypatch.setattr(seg_mod.threading, "Thread", _SyncThread)
    skipped = []
    frame = VideoFrame.new(data=np.zeros((4, 4, 3), dtype=np.uint8), format=VideoDataFormat.BGR)
    run_segmenter.run(
        seg_mod.ObjectSegmenterInputs(
            video=_FakeRecv([frame, None]),
            prompts=_FakeRecv([None]),
        ),
        seg_mod.ObjectSegmenterOutputs(
            segmentations=SimpleNamespace(send=lambda value: skipped.append(value)),
            video=SimpleNamespace(send=lambda value: skipped.append(value)),
        ),
    )
    assert skipped == []

    run_segmenter2 = seg_mod.ObjectSegmenter(seg_mod.ObjectSegmenterConfig())
    run_segmenter2._lock = contextlib.nullcontext()
    run_segmenter2._model = SimpleNamespace(set_classes=lambda prompts: None)
    infer_results = iter(
        [
            None,
            (
                np.ones((1, 2, 2), dtype=bool),
                np.array([[0, 0, 1, 1]], dtype=np.float32),
                np.array([0.8], dtype=np.float32),
                np.array([3], dtype=np.int64),
                ("cat",),
            ),
        ]
    )
    monkeypatch.setattr(run_segmenter2, "_infer", lambda rgb: next(infer_results))
    sent_seg = []
    sent_video = []
    run_segmenter2.run(
        seg_mod.ObjectSegmenterInputs(
            video=_FakeRecv([frame, frame, None]),
            prompts=_FakeRecv([TextFrame.new(text="cat"), None]),
        ),
        seg_mod.ObjectSegmenterOutputs(
            segmentations=SimpleNamespace(send=lambda value: sent_seg.append(value)),
            video=SimpleNamespace(send=lambda value: sent_video.append(value)),
        ),
    )
    assert len(sent_seg) == 1
    assert len(sent_video) == 2


def test_discord_paths(monkeypatch) -> None:
    import src.core.conduit.discord as discord_mod

    discord_mod._discord_bot = None
    discord_mod._discord_loop = None
    discord_mod._discord_thread = None
    discord_mod._discord_running = False
    discord_mod._active_discord_io = None
    discord_mod._voice_clients.clear()
    discord_mod._rings.clear()
    discord_mod._buffer.clear()
    discord_mod._playback_tasks.clear()

    monkeypatch.setattr(discord_mod.os, "getenv", lambda key: "")
    with pytest.raises(ValueError):
        discord_mod.DiscordIO(discord_mod.DiscordConfig())

    class _FakeBot:
        def __init__(self, intents):
            self.intents = intents
            self.user = "bot-user"

        async def start(self, token):
            self.token = token
            raise RuntimeError("bot boom")

    class _FakeLoop:
        def run_until_complete(self, coro):
            asyncio.run(coro)

    register_calls = []
    real_register_handlers = discord_mod.DiscordIO._register_handlers_for_bot
    real_thread = threading.Thread
    monkeypatch.setattr(discord_mod.os, "getenv", lambda key: "token")
    monkeypatch.setattr(discord_mod.discord.Intents, "default", lambda: "intents")
    monkeypatch.setattr(discord_mod.discord, "Bot", _FakeBot)
    monkeypatch.setattr(discord_mod.asyncio, "new_event_loop", lambda: _FakeLoop())
    monkeypatch.setattr(discord_mod.asyncio, "set_event_loop", lambda loop: None)
    monkeypatch.setattr(discord_mod.threading, "Thread", _SyncThread)
    monkeypatch.setattr(
        discord_mod.DiscordIO,
        "_register_handlers_for_bot",
        lambda self, bot: register_calls.append(bot),
    )
    io = discord_mod.DiscordIO(discord_mod.DiscordConfig(guild_ids=[1]))
    assert register_calls and register_calls[0].token == "token"
    discord_mod._discord_running = True
    io._ensure_discord_running()

    monkeypatch.setattr(discord_mod.DiscordIO, "_register_handlers_for_bot", real_register_handlers)
    monkeypatch.setattr(discord_mod.DiscordIO, "_ensure_discord_running", lambda self: None)
    monkeypatch.setattr(discord_mod.threading, "Thread", real_thread)
    io = discord_mod.DiscordIO(discord_mod.DiscordConfig(audio_buffer_seconds=2))

    class _SlashBot:
        def __init__(self):
            self.user = "user"
            self.events = {}
            self.commands = {}

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

    monkeypatch.setattr(discord_mod.discord, "Member", _Member)
    bot = _SlashBot()
    io._register_handlers_for_bot(bot)
    assert set(bot.commands) == {"join", "leave"}
    asyncio.run(bot.events["on_ready"]())

    class _Followup:
        def __init__(self):
            self.messages = []

        async def send(self, text):
            self.messages.append(text)

    class _Ctx:
        def __init__(self, author, guild):
            self.author = author
            self.guild = guild
            self.followup = _Followup()
            self.responses = []
            self.deferred = False

        async def respond(self, text):
            self.responses.append(text)

        async def defer(self):
            self.deferred = True

    asyncio.run(bot.commands["join"](_Ctx(author=object(), guild=None)))
    assert bot.commands["join"]

    missing_voice_ctx = _Ctx(author=_Member(voice=None), guild=None)
    asyncio.run(bot.commands["join"](missing_voice_ctx))
    assert missing_voice_ctx.responses == ["Join a voice channel first"]

    no_guild_ctx = _Ctx(author=_Member(voice=SimpleNamespace(channel=object())), guild=None)
    asyncio.run(bot.commands["join"](no_guild_ctx))
    assert no_guild_ctx.responses == ["Must be used in a guild"]

    no_channel_ctx = _Ctx(author=_Member(voice=SimpleNamespace(channel=None)), guild=SimpleNamespace(id=4))
    asyncio.run(bot.commands["join"](no_channel_ctx))
    assert no_channel_ctx.responses == ["Join a voice channel first"]

    create_task_calls = []

    class _Task:
        def __init__(self, done_state=False):
            self._done_state = done_state
            self.cancelled = False

        def done(self):
            return self._done_state

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(
        discord_mod.asyncio,
        "create_task",
        lambda coro: (coro.close(), create_task_calls.append("task"), _Task())[2],
    )

    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(discord_mod.asyncio, "sleep", _sleep)

    class _ExistingVC:
        def __init__(self, guild):
            self.guild = guild
            self.cleaned = False
            self.recording = False

        async def disconnect(self, force=True):
            return None

        def cleanup(self):
            self.cleaned = True

    class _NewVC:
        def __init__(self):
            self._connected = threading.Event()
            self._connected.set()
            self.recording = True
            self.stop_calls = 0
            self.disconnect_calls = 0
            self.started = False
            self.played = None

        def is_connected(self):
            return True

        def start_recording(self, sink, callback):
            self.started = True
            asyncio.get_running_loop().create_task(callback(sink))

        def stop_recording(self):
            self.stop_calls += 1

        async def disconnect(self, force=True):
            self.disconnect_calls += 1

        def play(self, source):
            self.played = source

    class _Channel:
        def __init__(self, result):
            self.result = result

        async def connect(self, timeout=30.0, reconnect=False):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    class _Guild:
        def __init__(self, gid, voice_client=None):
            self.id = gid
            self.voice_client = voice_client

    existing_guild = _Guild(10)
    existing_vc = _ExistingVC(existing_guild)
    existing_guild.voice_client = existing_vc
    new_vc = _NewVC()
    success_ctx = _Ctx(
        author=_Member(voice=SimpleNamespace(channel=_Channel(new_vc))),
        guild=existing_guild,
    )
    discord_mod._active_discord_io = io
    asyncio.run(bot.commands["join"](success_ctx))
    assert success_ctx.deferred is True
    assert success_ctx.followup.messages == ["Connected"]
    assert existing_vc.cleaned is True
    assert discord_mod._voice_clients[10] is new_vc
    assert discord_mod._rings[10].maxlen == io.max_frames
    assert create_task_calls == ["task"]

    discord_mod._playback_tasks[10] = _Task()
    leave_ctx = _Ctx(author=_Member(voice=SimpleNamespace(channel=None)), guild=SimpleNamespace(id=10))
    asyncio.run(bot.commands["leave"](leave_ctx))
    assert leave_ctx.responses == ["Disconnected"]
    assert new_vc.stop_calls == 1
    assert new_vc.disconnect_calls == 1
    assert discord_mod._playback_tasks == {}

    timeout_ctx = _Ctx(
        author=_Member(
            voice=SimpleNamespace(
                channel=_Channel(
                    SimpleNamespace(
                        _connected=threading.Event(),
                        recording=False,
                        is_connected=lambda: False,
                        start_recording=lambda sink, callback: None,
                        disconnect=lambda force=True: _sleep(0),
                    )
                )
            )
        ),
        guild=_Guild(11),
    )
    asyncio.run(bot.commands["join"](timeout_ctx))
    assert "Failed to join voice channel" in timeout_ctx.followup.messages[0]

    err4017_ctx = _Ctx(
        author=_Member(voice=SimpleNamespace(channel=_Channel(RuntimeError("4017 nope")))),
        guild=_Guild(12),
    )
    asyncio.run(bot.commands["join"](err4017_ctx))
    assert "DAVE protocol" in err4017_ctx.followup.messages[0]

    generic_err_ctx = _Ctx(
        author=_Member(voice=SimpleNamespace(channel=_Channel(RuntimeError("boom")))),
        guild=_Guild(13),
    )
    asyncio.run(bot.commands["join"](generic_err_ctx))
    assert generic_err_ctx.followup.messages == ["Failed to join voice channel: boom"]

    leave_no_guild_ctx = _Ctx(author=_Member(voice=None), guild=None)
    asyncio.run(bot.commands["leave"](leave_no_guild_ctx))
    assert leave_no_guild_ctx.responses == ["Must be used in a guild"]

    play_vc_break = SimpleNamespace(
        played=None,
        is_connected=lambda: True,
        play=lambda source: setattr(play_vc_break, "played", source),
    )
    discord_mod._voice_clients[20] = play_vc_break
    discord_mod._buffer[20] = deque([b"a" * 10])
    discord_mod._discord_running = False
    asyncio.run(io._playback_loop(20))
    assert play_vc_break.played is not None

    play_states = iter([True, False])
    play_vc_loop = SimpleNamespace(
        played=None,
        is_connected=lambda: next(play_states),
        play=lambda source: setattr(play_vc_loop, "played", source),
    )
    discord_mod._voice_clients[21] = play_vc_loop
    discord_mod._buffer[21] = deque([b"a" * 10])
    discord_mod._discord_running = True
    asyncio.run(io._playback_loop(21))
    assert play_vc_loop.played is not None

    discord_mod._buffer.clear()
    discord_mod._buffer[1] = deque([b"old"])
    discord_mod._buffer[2] = deque([b"old2"])
    monkeypatch.setattr(discord_mod.threading, "Thread", _SyncThread)
    pcm = AudioFrame.new(
        data=np.ones((2, 200), dtype=np.float32),
        sample_rate=16000,
        channels=2,
    )
    io.run(
        discord_mod.DiscordInputs(
            audio=_FakeRecv([pcm, None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="clear"), None]),
        ),
        discord_mod.DiscordOutputs(audio=SimpleNamespace(send=lambda value: None)),
    )
    assert len(discord_mod._buffer[1]) == 1
    assert len(discord_mod._buffer[2]) == 1

    captured = []
    io._output_audio = SimpleNamespace(send=lambda value: captured.append(value))
    discord_mod._active_discord_io = io
    discord_mod._DiscordSink.write(
        object.__new__(discord_mod._DiscordSink),
        b"\x01\x02\x03\x04",
        object(),
    )
    assert len(captured) == 1

    silent = discord_mod._DiscordAudioSource(deque())
    assert len(silent.read()) == 3840

    buffered = discord_mod._DiscordAudioSource(deque([b"a" * 1000, b"b" * 3000]))
    first = buffered.read()
    second = buffered.read()
    assert len(first) == 3840
    assert len(second) == 3840
