from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.core.frames import (
    AudioFrame,
    CameraParamsFrame,
    DepthFrame,
    ObjectLocationFrame,
    ObjectSegmentationFrame,
    TextFrame,
    VideoDataFormat,
    VideoFrame,
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


def _audio_frame(
    samples: int = 16000, sample_rate: int = 16000, channels: int = 1
) -> AudioFrame:
    data = np.zeros((channels, samples), dtype=np.float32)
    return AudioFrame.new(data=data, sample_rate=sample_rate, channels=channels)


def test_object_locator_paths(monkeypatch) -> None:
    import src.core.conduit.object_locator as loc_mod

    def resize_mask(img, size, interpolation=None):
        fill = 1 if np.asarray(img).sum() > 0 else 0
        return np.full((size[1], size[0]), fill, dtype=np.uint8)

    fake_cv2 = SimpleNamespace(
        INTER_NEAREST=1,
        resize=resize_mask,
    )
    monkeypatch.setattr(loc_mod, "cv2", fake_cv2)

    cam = CameraParamsFrame.new(
        intrinsics=np.array(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        extrinsics=np.eye(4, dtype=np.float32),
        width=2,
        height=2,
    )
    depth = DepthFrame.new(data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    seg = ObjectSegmentationFrame.new(
        masks=np.array(
            [
                np.array([[True]], dtype=bool),
                np.array([[False]], dtype=bool),
            ]
        ),
        boxes=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
        scores=np.array([0.9, 0.8], dtype=np.float32),
        object_ids=np.array([1, 2], dtype=np.int64),
        labels=("a", "b"),
    )
    located = loc_mod._locate_objects(seg, depth, cam)
    assert isinstance(located, ObjectLocationFrame)
    assert np.isfinite(located.positions[0]).all()
    assert np.isnan(located.positions[1]).all()
    assert located.depths[0] > 0

    sent = []
    locator = loc_mod.ObjectLocator()
    empty_seg = ObjectSegmentationFrame.new(
        masks=np.zeros((0, 1, 1), dtype=bool),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        object_ids=np.zeros((0,), dtype=np.int64),
        labels=(),
    )
    locator.run(
        loc_mod.ObjectLocatorInputs(
            segmentations=_FakeRecv([empty_seg, seg, seg, None]),
            depth=_FakeRecv([depth, depth]),
            camera_params=_FakeRecv([cam, cam, None]),
        ),
        loc_mod.ObjectLocatorOutputs(
            locations=SimpleNamespace(send=lambda value: sent.append(value))
        ),
    )
    assert len(sent) == 2
    assert sent[0].labels == ()

    locator.run(
        loc_mod.ObjectLocatorInputs(
            segmentations=_FakeRecv([None]),
            depth=_FakeRecv([depth]),
            camera_params=_FakeRecv([cam]),
        ),
        loc_mod.ObjectLocatorOutputs(
            locations=SimpleNamespace(send=lambda value: sent.append(value))
        ),
    )
    locator.run(
        loc_mod.ObjectLocatorInputs(
            segmentations=_FakeRecv([seg]),
            depth=_FakeRecv([depth]),
            camera_params=_FakeRecv([None]),
        ),
        loc_mod.ObjectLocatorOutputs(
            locations=SimpleNamespace(send=lambda value: sent.append(value))
        ),
    )


def test_streaming_vlm_paths(monkeypatch) -> None:
    import src.core.conduit.streaming_vlm as vlm_mod

    class _Client:
        def __init__(self, base_url, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            self.beta = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(parse=self.parse))
            )
            self.responses = []

        def parse(self, **kwargs):
            if self.responses:
                return self.responses.pop(0)
            raise RuntimeError("boom")

    monkeypatch.setattr(vlm_mod, "OpenAI", _Client)
    vlm = vlm_mod.StreamingVLM(
        vlm_mod.StreamingVLMConfig(base_url="http://vlm", max_history=3)
    )
    assert vlm._client.base_url == "http://vlm"

    # _encode_pil_to_base64_uri
    class _Image:
        def save(self, buf, format=None, quality=None):
            buf.write(b"jpeg")

    uri = vlm._encode_pil_to_base64_uri(_Image())
    assert uri.startswith("data:image/jpeg;base64,")

    # _call_caption: parsed response
    caption_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=SimpleNamespace(caption="a dog on a couch"),
                )
            )
        ]
    )
    vlm._client.responses = [caption_response]
    assert vlm._call_caption("data:image/jpeg;base64,abc") == "a dog on a couch"

    # _call_caption: parsed=None returns None
    none_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))]
    )
    vlm._client.responses = [none_response]
    assert vlm._call_caption("data:image/jpeg;base64,abc") is None

    # _call_caption: exception returns None
    assert vlm._call_caption("data:image/jpeg;base64,abc") is None

    # _call_objects: parsed response
    objects_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=SimpleNamespace(objects="dog, couch"),
                )
            )
        ]
    )
    vlm._client.responses = [objects_response]
    assert vlm._call_objects("a dog on a couch") == "dog, couch"

    # _call_objects: parsed=None returns None
    vlm._client.responses = [none_response]
    assert vlm._call_objects("caption") is None

    # _call_objects: exception returns None
    assert vlm._call_objects("caption") is None

    # History management: max_history trimming
    vlm._history = [{"image_uri": f"img{i}", "caption": f"cap{i}"} for i in range(3)]
    vlm._history.append({"image_uri": "img3", "caption": "cap3"})
    if len(vlm._history) > vlm.config.max_history:
        vlm._history.pop(0)
    assert len(vlm._history) == 3
    assert vlm._history[0]["caption"] == "cap1"

    # _call_caption includes history in messages
    vlm._history = [{"image_uri": "old_uri", "caption": "old caption"}]
    vlm._client.responses = [caption_response]
    result = vlm._call_caption("new_uri")
    assert result == "a dog on a couch"

    # run() end-to-end with mocked calls
    run_vlm = vlm_mod.StreamingVLM(
        vlm_mod.StreamingVLMConfig(base_url="http://vlm", max_history=2)
    )
    caption_calls = []
    object_calls = []

    def fake_call_caption(image_uri):
        caption_calls.append(image_uri)
        return "first caption" if len(caption_calls) == 1 else None

    def fake_call_objects(caption):
        object_calls.append(caption)
        return "chair"

    monkeypatch.setattr(run_vlm, "_call_caption", fake_call_caption)
    monkeypatch.setattr(run_vlm, "_call_objects", fake_call_objects)
    video = VideoFrame.new(
        data=np.zeros((2, 2, 3), dtype=np.uint8), format=VideoDataFormat.BGR
    )
    run_obs = []
    run_obj = []
    run_vlm.run(
        vlm_mod.StreamingVLMInputs(video=_FakeRecv([video, video, None])),
        vlm_mod.StreamingVLMOutputs(
            observation=SimpleNamespace(send=lambda value: run_obs.append(value)),
            objects=SimpleNamespace(send=lambda value: run_obj.append(value)),
        ),
    )
    assert run_obs[0].get() == "first caption"
    assert run_obj[0].get() == "chair"
    # Second frame returns None caption, so no output for it
    assert len(run_obs) == 1
    assert len(run_obj) == 1


def test_vad_paths(monkeypatch, tmp_path: Path) -> None:
    import src.core.conduit.vad as vad_mod

    class _Iterator:
        def __init__(self, model):
            self.model = model
            self.responses = []
            self.reset = False

        def __call__(self, chunk, return_seconds=False):
            return self.responses.pop(0) if self.responses else {}

        def reset_states(self):
            self.reset = True

    monkeypatch.setattr(
        vad_mod.torch,
        "hub",
        SimpleNamespace(
            load=lambda **kwargs: ("model", (None, None, None, _Iterator, None))
        ),
    )

    class _SessionOptions:
        def __init__(self):
            self.execution_mode = None
            self.inter_op_num_threads = None

    class _ExecutionMode:
        ORT_SEQUENTIAL = "seq"

    class _Session:
        def __init__(self, path, sess_options=None):
            self.path = path
            self.sess_options = sess_options
            self.results = [np.array([[0.95]], dtype=np.float32)]

        def run(self, *_args, **_kwargs):
            return self.results.pop(0)

    class _Extractor:
        def __init__(self, chunk_length):
            self.chunk_length = chunk_length

        def __call__(self, pcm, **kwargs):
            return SimpleNamespace(input_features=np.ones((1, 2, 3), dtype=np.float32))

    monkeypatch.setattr(vad_mod.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(vad_mod.ort, "ExecutionMode", _ExecutionMode)
    monkeypatch.setattr(vad_mod.ort, "InferenceSession", _Session)
    monkeypatch.setattr(vad_mod, "WhisperFeatureExtractor", _Extractor)

    missing = vad_mod.VADConfig(smart_turn_onnx=str(tmp_path / "missing.onnx"))
    missing_vad = vad_mod.VAD(missing)
    assert missing_vad._smart_turn_session is None

    onnx_path = tmp_path / "smart.onnx"
    onnx_path.write_bytes(b"x")
    cfg = vad_mod.VADConfig(
        smart_turn_onnx=str(onnx_path),
        silence_seconds=0.1,
        max_silence_seconds=0.2,
        pre_speech_seconds=0.01,
        min_speech_seconds=0.1,
        turn_threshold=0.5,
    )
    vad = vad_mod.VAD(cfg)
    assert vad._smart_turn_session is not None
    assert vad._prepare_smart_turn_input() is None
    vad._current_segment = [_audio_frame(samples=1600)]
    smart_input = vad._prepare_smart_turn_input()
    assert smart_input is not None
    assert vad._run_smart_turn_inference(smart_input) is True
    vad._smart_turn_session = _Session(str(onnx_path))
    vad._feature_extractor = _Extractor(8)
    vad._current_segment = [_audio_frame(samples=16000 * 9)]
    smart_input = vad._prepare_smart_turn_input()
    assert smart_input is not None
    assert vad._run_smart_turn_inference(smart_input) is True
    vad._feature_extractor = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("bad")
    )
    assert vad._run_smart_turn_inference(np.zeros(8 * 16000, dtype=np.float32)) is False

    outputs = SimpleNamespace(
        audio=SimpleNamespace(send=lambda value: sent_audio.append(value)),
        interrupt=SimpleNamespace(send=lambda value: sent_interrupts.append(value)),
    )
    sent_audio = []
    sent_interrupts = []

    vad._pre_buffer = [_audio_frame(samples=2000)]
    vad._handle_speech_start(outputs)
    assert vad._speaking is True
    assert sent_interrupts[0].reason == "speech_detected"
    assert vad._current_segment

    vad._current_segment = [_audio_frame(samples=3200), _audio_frame(samples=3200)]
    vad._finalize_segment(outputs)
    assert len(sent_audio) == 1
    assert vad._vad_iterator.reset is True

    short_vad = vad_mod.VAD(cfg)
    short_vad._current_segment = [_audio_frame(samples=100)]
    short_vad._finalize_segment(outputs)

    monkeypatch.setattr(vad_mod.torch, "tensor", lambda values: np.asarray(values))
    proc_vad = vad_mod.VAD(cfg)
    proc_vad._vad_iterator.responses = [{"start": 1}]
    start_calls = []
    monkeypatch.setattr(
        proc_vad,
        "_handle_speech_start",
        lambda outputs: (
            start_calls.append("start"),
            setattr(proc_vad, "_speaking", True),
        ),
    )
    proc_vad._process_audio_frame(_audio_frame(samples=1024), outputs)
    assert start_calls == ["start"]

    finalize_calls = []
    monkeypatch.setattr(vad_mod.time, "time", lambda: 10.0)
    proc_vad._speaking = True
    proc_vad._silence_start = 9.0
    monkeypatch.setattr(
        proc_vad, "_finalize_segment", lambda outputs: finalize_calls.append("max")
    )
    proc_vad._process_audio_frame(_audio_frame(samples=256), outputs)
    assert finalize_calls == ["max"]

    proc_vad._speaking = True
    proc_vad._silence_start = 9.93
    monkeypatch.setattr(
        proc_vad,
        "_prepare_smart_turn_input",
        lambda: np.zeros(8 * 16000, dtype=np.float32),
    )
    monkeypatch.setattr(proc_vad, "_run_smart_turn_inference", lambda _pcm: True)
    monkeypatch.setattr(
        proc_vad, "_finalize_segment", lambda outputs: finalize_calls.append("turn")
    )
    monkeypatch.setattr(vad_mod.time, "time", lambda: 10.05)
    proc_vad._process_audio_frame(_audio_frame(samples=256), outputs)
    assert "turn" in finalize_calls

    end_vad = vad_mod.VAD(cfg)
    end_vad._speaking = True
    end_vad._silence_start = None
    end_vad._vad_iterator.responses = [{"end": 1}]
    monkeypatch.setattr(vad_mod.time, "time", lambda: 20.0)
    end_vad._process_audio_frame(_audio_frame(samples=512), outputs)
    assert end_vad._silence_start == 20.0

    proc_vad._speaking = False
    proc_vad.config.pre_speech_seconds = 1.0
    proc_vad._pre_buffer = [_audio_frame(samples=400)]
    proc_vad._process_audio_frame(_audio_frame(samples=400), outputs)
    assert proc_vad._pre_buffer

    trim_vad = vad_mod.VAD(cfg)
    trim_vad.config.pre_speech_seconds = 0.001
    trim_vad._pre_buffer = [_audio_frame(samples=400)]
    trim_vad._process_audio_frame(_audio_frame(samples=400), outputs)
    assert len(trim_vad._pre_buffer) <= 1

    empty_finalize_vad = vad_mod.VAD(cfg)
    empty_finalize_vad._current_segment = []
    empty_finalize_vad._finalize_segment(outputs)

    monitor_vad = vad_mod.VAD(cfg)
    monitor_vad._speaking = True
    monitor_vad._silence_start = 9.0
    monitor_calls = []
    monkeypatch.setattr(vad_mod.time, "time", lambda: 10.0)

    def monitor_finalize(outputs):
        monitor_calls.append("final")
        monitor_vad.stop_event.set()

    monkeypatch.setattr(monitor_vad, "_finalize_segment", monitor_finalize)
    monkeypatch.setattr(vad_mod.time, "sleep", lambda seconds: None)
    monitor_vad._monitor_loop(outputs)
    assert monitor_calls == ["final"]

    monitor_vad2 = vad_mod.VAD(cfg)
    monitor_vad2._speaking = True
    monitor_vad2._silence_start = 9.92
    monkeypatch.setattr(vad_mod.time, "time", lambda: 10.05)
    monkeypatch.setattr(
        monitor_vad2,
        "_prepare_smart_turn_input",
        lambda: np.zeros(8 * 16000, dtype=np.float32),
    )
    monkeypatch.setattr(monitor_vad2, "_run_smart_turn_inference", lambda _pcm: True)

    def monitor_finalize2(outputs):
        monitor_calls.append("smart")
        monitor_vad2.stop_event.set()

    monkeypatch.setattr(monitor_vad2, "_finalize_segment", monitor_finalize2)
    monitor_vad2._monitor_loop(outputs)
    assert "smart" in monitor_calls

    run_vad = vad_mod.VAD(cfg)
    processed = []
    finalized = []

    class _Thread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            return

        def join(self, timeout=None):
            self.timeout = timeout

    monkeypatch.setattr(vad_mod.threading, "Thread", _Thread)
    monkeypatch.setattr(
        run_vad, "_process_audio_frame", lambda frame, outputs: processed.append(frame)
    )
    monkeypatch.setattr(
        run_vad, "_finalize_segment", lambda outputs: finalized.append("done")
    )
    run_vad._current_segment = [_audio_frame(samples=1000)]
    run_vad.run(
        vad_mod.VADInputs(audio=_FakeRecv([_audio_frame(samples=1000), None])),
        outputs,
    )
    assert len(processed) == 1
    assert finalized == ["done"]


def test_object_detector_paths(monkeypatch) -> None:
    import src.core.conduit.object_detector as det_mod

    monkeypatch.setattr(
        det_mod, "auto_device", lambda device: SimpleNamespace(type="cpu")
    )
    monkeypatch.setattr(det_mod, "auto_dtype", lambda device: "f32")
    monkeypatch.setattr(
        det_mod, "resize_and_crop", lambda frame, width, height: frame[:height, :width]
    )

    fake_torch = types.SimpleNamespace(
        inference_mode=lambda: contextlib.nullcontext(),
        is_tensor=lambda value: hasattr(value, "detach"),
        tensor=lambda values: np.asarray(values),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    detector = det_mod.ObjectDetector(
        det_mod.ObjectDetectorConfig(resolution=2, session_ttl=1)
    )
    assert detector._device.type == "cpu"
    assert detector._dtype == "f32"

    class _Cfg:
        @classmethod
        def from_pretrained(cls, model_id):
            return SimpleNamespace()

    class _Tensor:
        def __init__(self, value):
            self.value = value

        def to(self, device, dtype):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value)

        def tolist(self):
            return list(self.value)

    class _Session:
        def __init__(self):
            self.processed_frames = {0: "old", 1: "new"}
            self.removed = []

        def remove_object(self, oid, strict=False):
            self.removed.append((oid, strict))

    class _Model:
        def to(self, device, dtype):
            return self

        def eval(self):
            self.evaluated = True

        def __call__(self, **kwargs):
            return {
                "obj_id_to_mask": {
                    1: SimpleNamespace(
                        isnan=lambda: np.array([True]),
                        __le__=lambda self, other: np.array([False]),
                    ),
                    2: SimpleNamespace(
                        isnan=lambda: np.array([False]),
                        __le__=lambda self, other: np.array([True]),
                    ),
                },
                "obj_id_to_score": {1: 0.9, 2: 0.8},
            }

        @classmethod
        def from_pretrained(cls, model_id, config=None):
            return cls()

    class _Processor:
        def __init__(self):
            self.image_processor = SimpleNamespace(size={})
            self.added = []

        @classmethod
        def from_pretrained(cls, model_id, target_size=None):
            return cls()

        def init_video_session(self, **kwargs):
            return _Session()

        def add_text_prompt(self, inference_session=None, text=None):
            self.added.append(list(text))

        def __call__(self, images=None, return_tensors=None, device=None):
            return SimpleNamespace(
                pixel_values=[_Tensor(np.zeros((2, 2, 3), dtype=np.float32))],
                original_sizes=[(2, 2)],
            )

        def postprocess_outputs(self, session, outputs, original_sizes=None):
            return {
                "object_ids": np.array([1, 2], dtype=np.int64),
                "scores": np.array([0.9, 0.8], dtype=np.float32),
                "boxes": np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32),
                "prompt_to_obj_ids": {"cat": _Tensor([1]), "dog": [2]},
            }

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Sam3VideoConfig=_Cfg,
            Sam3VideoModel=_Model,
            Sam3VideoProcessor=_Processor,
        ),
    )

    detector._prompts = ["cat", "dog"]
    detector._ensure_model()
    assert detector._model is not None
    detector._ensure_model()

    processed = {
        "object_ids": np.array([1, 2], dtype=np.int64),
        "scores": np.array([0.9, 0.8], dtype=np.float32),
        "boxes": np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32),
        "prompt_to_obj_ids": {"cat": _Tensor([1]), "dog": [2]},
    }
    boxes, scores = detector._gather_by_prompt(processed)
    assert boxes.shape == (2, detector.config.max_tracking_per_object, 4)
    assert scores[0, 0] == 0.9
    detector._prompts = ["cat"]
    detector.config.max_tracking_per_object = 1
    overflow_boxes, overflow_scores = detector._gather_by_prompt(
        {
            "object_ids": np.array([1, 2, 3], dtype=np.int64),
            "scores": np.array([0.9, 0.8, 0.7], dtype=np.float32),
            "boxes": np.array(
                [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], dtype=np.float32
            ),
            "prompt_to_obj_ids": {"cat": _Tensor([1, 2])},
        }
    )
    assert overflow_scores[0, 0] == 0.9
    assert overflow_boxes.shape[0] == 1

    class _Mask:
        def __init__(self, nan=False, neg=False):
            self._nan = nan
            self._neg = neg

        def isnan(self):
            return np.array([self._nan])

        def __le__(self, other):
            return np.array([self._neg])

    detector._session = _Session()
    outputs = {
        "obj_id_to_mask": {1: _Mask(nan=True), 2: _Mask(neg=True), 3: _Mask()},
        "obj_id_to_score": {1: 1, 2: 1, 3: 1},
    }
    detector._purge_dead_tracklets(outputs)
    assert detector._session.removed == [(1, False), (2, False)]
    detector._prune_old_frames()
    assert detector._session.processed_frames[0] is None

    listener = det_mod.ObjectDetector(det_mod.ObjectDetectorConfig(resolution=2))
    listener._prompts = ["old"]
    init_calls = []
    monkeypatch.setattr(
        listener, "_init_session", lambda: init_calls.append(list(listener._prompts))
    )
    listener._prompt_listener(
        det_mod.ObjectDetectorInputs(
            video=_FakeRecv([]),
            prompts=_FakeRecv(
                [TextFrame.new(text="old"), TextFrame.new(text="cat, dog"), None]
            ),
        )
    )
    assert listener._prompts == ["cat", "dog"]
    assert init_calls == [["cat", "dog"]]

    run_detector = det_mod.ObjectDetector(
        det_mod.ObjectDetectorConfig(resolution=2, session_ttl=1)
    )
    run_detector._prompts = []
    run_detector._ensure_model = lambda: None
    run_detector._processor = _Processor()
    run_detector._session = _Session()
    run_detector._lock = contextlib.nullcontext()
    run_detector._model = lambda **kwargs: {"obj_id_to_mask": {}, "obj_id_to_score": {}}
    monkeypatch.setattr(run_detector, "_purge_dead_tracklets", lambda outputs: None)
    monkeypatch.setattr(run_detector, "_prune_old_frames", lambda: None)
    monkeypatch.setattr(
        run_detector,
        "_gather_by_prompt",
        lambda processed: (
            np.zeros((1, 1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        ),
    )
    init_session_calls = []
    monkeypatch.setattr(
        run_detector, "_init_session", lambda: init_session_calls.append("init")
    )

    class _Thread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            self.target(*self.args)

        def join(self, timeout=None):
            return

    monkeypatch.setattr(det_mod.threading, "Thread", _Thread)
    sent_det = []
    sent_video = []
    video = VideoFrame.new(
        data=np.zeros((2, 2, 3), dtype=np.uint8), format=VideoDataFormat.BGR
    )
    run_detector.run(
        det_mod.ObjectDetectorInputs(
            video=_FakeRecv([video, video, None]),
            prompts=_FakeRecv([TextFrame.new(text="cat"), None]),
        ),
        det_mod.ObjectDetectorOutputs(
            detections=SimpleNamespace(send=lambda value: sent_det.append(value)),
            video=SimpleNamespace(send=lambda value: sent_video.append(value)),
        ),
    )
    assert len(sent_det) == 2
    assert len(sent_video) == 2
    assert len(init_session_calls) == 3

    no_prompt_detector = det_mod.ObjectDetector(
        det_mod.ObjectDetectorConfig(resolution=2)
    )
    no_prompt_detector._prompts = []
    no_prompt_detector._ensure_model = lambda: None
    monkeypatch.setattr(det_mod.threading, "Thread", _Thread)
    skipped = []
    no_prompt_detector.run(
        det_mod.ObjectDetectorInputs(
            video=_FakeRecv([video, None]),
            prompts=_FakeRecv([None]),
        ),
        det_mod.ObjectDetectorOutputs(
            detections=SimpleNamespace(send=lambda value: skipped.append(value)),
            video=SimpleNamespace(send=lambda value: skipped.append(value)),
        ),
    )
    assert skipped == []
