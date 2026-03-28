from __future__ import annotations

import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.frames import EOS, InterruptFrame, TextFrame


class _SyncThread:
    def __init__(self, target, args=(), daemon=False, **kwargs):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self):
        self.target(*self.args)

    def join(self, timeout=None):
        return None


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def test_qwen_tts_model_and_streaming_paths(monkeypatch, tmp_path: Path) -> None:
    # Prevent real CUDA usage — CI runners may have mismatched drivers
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    import transformers.utils.generic as transformers_generic

    if not hasattr(transformers_generic, "check_model_inputs"):
        monkeypatch.setattr(
            transformers_generic,
            "check_model_inputs",
            lambda *args, **kwargs: lambda func: func,
            raising=False,
        )

    import src.core.conduit.qwen_tts.model as model_mod
    import src.core.utils as utils_mod

    class _Processor:
        def __call__(self, text=None, return_tensors=None, padding=None):
            if text == "single":
                ids = torch.tensor([1, 2], dtype=torch.long)
            else:
                ids = torch.tensor([[3, 4]], dtype=torch.long)
            return {"input_ids": ids}

    class _SpeechTokenizer:
        def __init__(self):
            self.calls = []
            self.model = SimpleNamespace(
                decoder=lambda codes: torch.tensor(
                    [[0.1, 0.2, 0.3]], dtype=torch.float32
                )
            )

        def encode(self, wavs, sr=None):
            self.calls.append((wavs, sr))
            if isinstance(wavs, list):
                return SimpleNamespace(
                    audio_codes=[torch.tensor([1, 2], dtype=torch.long) for _ in wavs]
                )
            return SimpleNamespace(audio_codes=[torch.tensor([3, 4], dtype=torch.long)])

        def get_output_sample_rate(self):
            return 24000

    class _FakeQwenModel:
        def __init__(self):
            self.generate_config = {"top_k": 7}
            self.tts_model_type = "base"
            self.speech_tokenizer = _SpeechTokenizer()
            self.speaker_encoder_sample_rate = 16000
            self.generated = None
            self.talker = SimpleNamespace()

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def extract_speaker_embedding(self, audio, sr):
            return torch.tensor([float(len(audio)), float(sr)], dtype=torch.float32)

        def generate(self, **kwargs):
            self.generated = kwargs
            return "generated"

    fake_model = _FakeQwenModel()
    processor = _Processor()
    qwen = model_mod.Qwen3TTSModel(fake_model, processor)
    monkeypatch.setattr(model_mod, "Qwen3TTSForConditionalGeneration", _FakeQwenModel)

    register_calls = []
    monkeypatch.setattr(
        model_mod.AutoConfig,
        "register",
        lambda name, cls: register_calls.append(("config", name)),
    )
    monkeypatch.setattr(
        model_mod.AutoModel,
        "register",
        lambda cfg, cls: register_calls.append(("model", cfg)),
    )
    monkeypatch.setattr(
        model_mod.AutoProcessor,
        "register",
        lambda cfg, cls: register_calls.append(("processor", cfg)),
    )
    monkeypatch.setattr(
        model_mod.AutoModel, "from_pretrained", lambda path, **kwargs: fake_model
    )
    monkeypatch.setattr(
        model_mod.AutoProcessor,
        "from_pretrained",
        lambda path, fix_mistral_regex=True: processor,
    )
    wrapped = model_mod.Qwen3TTSModel.from_pretrained("demo")
    assert wrapped.model is fake_model
    assert len(register_calls) == 3

    monkeypatch.setattr(
        model_mod.AutoModel, "from_pretrained", lambda path, **kwargs: object()
    )
    with pytest.raises(TypeError):
        model_mod.Qwen3TTSModel.from_pretrained("demo")
    monkeypatch.setattr(
        model_mod.AutoModel, "from_pretrained", lambda path, **kwargs: fake_model
    )

    assert qwen._as_list(1) == [1]
    assert qwen._as_list([1]) == [1]
    assert "assistant" in qwen._assistant_text("hi")
    assert "assistant" in qwen._build_ref_text("ref")
    toks = qwen._tokenize(["single", "batched"])
    assert toks[0].shape[0] == 1
    assert toks[1].shape == (1, 2)

    monkeypatch.setattr(
        model_mod.librosa,
        "load",
        lambda path, sr=None, mono=True: (
            np.array([1.0, 2.0], dtype=np.float32),
            22050,
        ),
    )
    wav, sr = qwen._load_audio("demo.wav")
    assert sr == 22050
    wav2, sr2 = qwen._load_audio(
        (np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32), 16000)
    )
    assert wav2.dtype == np.float32 and sr2 == 16000

    items_dict = qwen._items_to_prompt(
        [
            model_mod.VoiceClonePromptItem(
                ref_code=torch.tensor([1]),
                ref_spk_embedding=torch.tensor([0.1]),
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text="a",
            )
        ]
    )
    assert items_dict["icl_mode"] == [True]
    defaults = qwen._gen_defaults(top_k=None, temperature=0.1)
    assert defaults["top_k"] == 7
    assert defaults["temperature"] == 0.1

    fake_model.tts_model_type = "chat"
    with pytest.raises(ValueError):
        qwen.create_voice_clone_prompt(
            ref_audio=(np.array([1.0], dtype=np.float32), 16000)
        )
    fake_model.tts_model_type = "base"
    with pytest.raises(ValueError):
        qwen.create_voice_clone_prompt(
            ref_audio=[
                (np.array([1.0], dtype=np.float32), 16000),
                (np.array([2.0], dtype=np.float32), 16000),
            ],
            ref_text=["one"],
            x_vector_only_mode=[False, False],
        )
    with pytest.raises(ValueError):
        qwen.create_voice_clone_prompt(
            ref_audio=(np.array([1.0], dtype=np.float32), 16000),
            ref_text=None,
            x_vector_only_mode=False,
        )

    monkeypatch.setattr(
        model_mod.librosa, "resample", lambda y, orig_sr, target_sr: np.asarray(y) * 2
    )
    same_sr_items = qwen.create_voice_clone_prompt(
        ref_audio=[
            (np.array([1.0, 2.0], dtype=np.float32), 16000),
            (np.array([3.0, 4.0], dtype=np.float32), 16000),
        ],
        ref_text=["hello", ""],
        x_vector_only_mode=[False, True],
    )
    assert len(same_sr_items) == 2

    diff_sr_items = qwen.create_voice_clone_prompt(
        ref_audio=[
            (np.array([1.0, 2.0], dtype=np.float32), 16000),
            (np.array([3.0, 4.0], dtype=np.float32), 8000),
        ],
        ref_text=["hello", "world"],
        x_vector_only_mode=[False, False],
    )
    assert len(diff_sr_items) == 2

    with pytest.raises(ValueError):
        qwen.generate_voice_clone("hello", voice_clone_prompt=None)
    with pytest.raises(ValueError):
        qwen.generate_voice_clone(
            ["a", "b"],
            voice_clone_prompt=[same_sr_items[0], same_sr_items[1], diff_sr_items[0]],
        )

    out = qwen.generate_voice_clone(
        ["a", "b"],
        language="English",
        voice_clone_prompt=[same_sr_items[0], same_sr_items[1]],
    )
    assert out == "generated"
    assert fake_model.generated["languages"] == ["English", "English"]
    assert fake_model.generated["non_streaming_mode"] is False

    monkeypatch.setattr(utils_mod, "auto_device", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr(utils_mod, "auto_dtype", lambda device: torch.float32)
    monkeypatch.setattr(model_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        model_mod.Qwen3TTSModel,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model=fake_model, processor=processor),
    )
    loaded = model_mod.SimpleStreamingTTS.load("demo", "cpu")
    assert loaded.sample_rate == 24000

    fake_cuda_device = SimpleNamespace(type="cuda", __str__=lambda self: "cuda")
    monkeypatch.setattr(utils_mod, "auto_device", lambda cfg: fake_cuda_device)
    monkeypatch.setattr(utils_mod, "auto_dtype", lambda device: torch.float16)
    monkeypatch.setattr(model_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        model_mod.torch, "set_float32_matmul_precision", lambda value: None
    )
    monkeypatch.setattr(
        model_mod.torch.cuda, "Stream", lambda: SimpleNamespace(synchronize=lambda: None)
    )
    loaded_cuda = model_mod.SimpleStreamingTTS.load("demo", "cuda")
    assert loaded_cuda._decode_stream is not None

    save_calls = []
    monkeypatch.setattr(
        model_mod.torch, "save", lambda prompt, path: save_calls.append((prompt, path))
    )
    voice_path = loaded.register_voice(
        "voice name", (np.array([1.0], dtype=np.float32), 16000), "hello"
    )
    assert save_calls and voice_path.name.startswith("voice_name")

    loaded._decode_stream = None
    audio = loaded._decode_audio([torch.tensor([1]), torch.tensor([2])])
    assert audio.ndim == 1

    class _Stream:
        def synchronize(self):
            self.synced = True

    loaded_cuda._decode_stream = _Stream()
    monkeypatch.setattr(
        model_mod.torch.cuda, "stream", lambda stream: contextlib.nullcontext()
    )
    loaded_cuda._decode_audio([torch.tensor([1]), torch.tensor([2])])

    streaming = model_mod.SimpleStreamingTTS(fake_model, processor, torch.device("cpu"))
    monkeypatch.setattr(
        model_mod,
        "_streaming_config",
        model_mod.StreamingConfig(min_initial_frames=1, yield_every_n_frames=1),
    )
    monkeypatch.setattr(
        model_mod.torch, "load", lambda path, weights_only=False: "prompt"
    )
    monkeypatch.setattr(model_mod.threading, "Thread", _SyncThread)
    monkeypatch.setattr(
        streaming,
        "_decode_audio",
        lambda codes: np.arange(len(codes) * 2, dtype=np.float32),
    )

    class _Talker:
        def __init__(self):
            self.calls = 0
            self.forward = self._forward

        def _forward(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                hidden_states=(None, torch.tensor([self.calls], dtype=torch.float32))
            )

    talker = _Talker()
    streaming._talker = talker
    streaming._model.generate_voice_clone = lambda **kwargs: [
        streaming._talker.forward(),
        streaming._talker.forward(),
    ]
    chunks = list(streaming.generate_streaming("hi", tmp_path / "voice.pt"))
    assert chunks[0][1]["frame_idx"] == 1
    assert chunks[-1][1]["is_final"] is True

    streaming_err = model_mod.SimpleStreamingTTS(
        fake_model, processor, torch.device("cpu")
    )
    streaming_err._talker = _Talker()
    monkeypatch.setattr(model_mod.threading, "Thread", _SyncThread)
    monkeypatch.setattr(
        model_mod.torch, "load", lambda path, weights_only=False: "prompt"
    )
    streaming_err._model.generate_voice_clone = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    with pytest.raises(RuntimeError):
        list(streaming_err.generate_streaming("hi", tmp_path / "voice.pt"))


def test_qwen_tts_component_paths(monkeypatch, tmp_path: Path) -> None:
    import transformers.utils.generic as transformers_generic

    if not hasattr(transformers_generic, "check_model_inputs"):
        monkeypatch.setattr(
            transformers_generic,
            "check_model_inputs",
            lambda *args, **kwargs: lambda func: func,
            raising=False,
        )

    import src.core.conduit.qwen_tts.component as comp_mod
    import src.core.conduit.qwen_tts.model as qwen_model_mod

    class _FakeFilter:
        def feed(self, text, force=False):
            if force:
                return "flush"
            return text

    monkeypatch.setattr(comp_mod, "StreamFilter", _FakeFilter)

    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "good.wav").write_bytes(b"x")
    (voices_dir / "good.txt").write_text("hello", encoding="utf-8")
    (voices_dir / "bad.wav").write_bytes(b"x")
    (voices_dir / "bad.txt").write_text("", encoding="utf-8")
    (voices_dir / "skip.mp3").write_bytes(b"x")

    options = comp_mod.QwenTTS.get_options({"ref_samples_dir": str(voices_dir)})
    assert (
        options["config"]["voice_id"][0]["value"] == "bad"
        or options["config"]["voice_id"][0]["value"] == "good"
    )
    assert (
        comp_mod.QwenTTS.get_options({"ref_samples_dir": str(tmp_path / "missing")})
        == {}
    )

    comp = comp_mod.QwenTTS(
        comp_mod.QwenTTSConfig(
            ref_samples_dir=str(voices_dir),
            voice_id="good",
            model_id="demo",
            language="English",
        )
    )

    fake_tts = SimpleNamespace(
        sample_rate=24000,
        register_voice=lambda name, path, transcript: Path(path).with_suffix(".pt"),
        generate_streaming=lambda text, voice_path, language: [
            (np.array([0.1, 0.2], dtype=np.float32), {}),
            (np.array([], dtype=np.float32), {"is_final": True}),
        ],
    )
    monkeypatch.setattr(
        qwen_model_mod,
        "SimpleStreamingTTS",
        SimpleNamespace(load=lambda model_id, device: fake_tts),
    )
    comp._load_model()
    assert comp._tts is fake_tts

    comp._register_voices()
    assert "good" in comp._voice_paths

    missing_comp = comp_mod.QwenTTS(
        comp_mod.QwenTTSConfig(ref_samples_dir=str(tmp_path / "missing"))
    )
    missing_comp._tts = fake_tts
    missing_comp._register_voices()
    assert missing_comp._voice_paths == {}

    failing_comp = comp_mod.QwenTTS(
        comp_mod.QwenTTSConfig(ref_samples_dir=str(voices_dir))
    )
    failing_comp._tts = SimpleNamespace(
        register_voice=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("bad")
        ),
    )
    failing_comp._register_voices()
    assert failing_comp._voice_paths == {}

    assert comp._resolve_voice().name == "good.pt"
    fallback_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    fallback_comp._voice_paths = {"x": Path("x.pt")}
    assert fallback_comp._resolve_voice().name == "x.pt"
    empty_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    with pytest.raises(RuntimeError):
        empty_comp._resolve_voice()

    audio_sent = []
    text_sent = []

    class _Queue:
        def __init__(self, owner):
            self.owner = owner
            self.items = [(1, "skip"), (0, "hello"), (0, "boom")]

        def get(self, timeout=None):
            if self.items:
                return self.items.pop(0)
            self.owner.stop_event.set()
            raise comp_mod.Empty

        def empty(self):
            return not self.items

        def put(self, item):
            self.items.append(item)

        def get_nowait(self):
            return self.items.pop(0)

    worker_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    worker_comp._tts = SimpleNamespace(
        sample_rate=22050,
        generate_streaming=lambda text, voice_path, language: (
            (_ for _ in ()).throw(RuntimeError("boom"))
            if text == "boom"
            else iter(
                [
                    (np.array([0.1], dtype=np.float32), {}),
                    (np.array([], dtype=np.float32), {"is_final": True}),
                ]
            )
        ),
    )
    worker_comp._voice_paths = {"good": Path("good.pt")}
    worker_comp._task_queue = _Queue(worker_comp)
    worker_comp._worker(
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda value: audio_sent.append(value)),
            text=SimpleNamespace(send=lambda value: text_sent.append(value)),
        )
    )
    assert len(audio_sent) == 1
    assert text_sent[0].text == "hello"

    run_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    run_comp._task_queue.put((0, "old"))
    calls = []
    monkeypatch.setattr(run_comp, "_load_model", lambda: calls.append("load"))
    monkeypatch.setattr(run_comp, "_register_voices", lambda: calls.append("voices"))
    monkeypatch.setattr(run_comp, "_worker", lambda outputs: calls.append("worker"))
    monkeypatch.setattr(comp_mod.threading, "Thread", _SyncThread)
    run_comp.run(
        comp_mod.QwenTTSInputs(
            text=_FakeRecv([TextFrame.new(text="hello"), EOS.END, None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="stop"), None]),
        ),
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda value: None),
            text=SimpleNamespace(send=lambda value: None),
        ),
    )
    assert calls == ["load", "voices", "worker"]
    queued = []
    while not run_comp._task_queue.empty():
        queued.append(run_comp._task_queue.get_nowait())
    assert queued[-2:] == [(1, "hello"), (1, "flush")]


def test_qwen_tts_component_remaining_branches(monkeypatch) -> None:
    import src.core.conduit.qwen_tts.component as comp_mod

    class _FakeFilter:
        def feed(self, text, force=False):
            return "flush" if force else text

    monkeypatch.setattr(comp_mod, "StreamFilter", _FakeFilter)

    worker_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    worker_comp._voice_paths = {"good": Path("good.pt")}
    mismatch_audio = []

    class _MismatchQueue:
        def __init__(self, owner):
            self.owner = owner
            self.items = [(0, "hello")]

        def get(self, timeout=None):
            if self.items:
                return self.items.pop(0)
            self.owner.stop_event.set()
            raise comp_mod.Empty

        def empty(self):
            return not self.items

    def _mismatch_stream(*args, **kwargs):
        yield np.array([0.1], dtype=np.float32), {}
        worker_comp._generation = 1
        yield np.array([0.2], dtype=np.float32), {}

    worker_comp._tts = SimpleNamespace(
        sample_rate=24000,
        generate_streaming=_mismatch_stream,
    )
    worker_comp._task_queue = _MismatchQueue(worker_comp)
    worker_comp._worker(
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda value: mismatch_audio.append(value)),
            text=SimpleNamespace(send=lambda value: None),
        )
    )
    assert len(mismatch_audio) == 1

    run_comp = comp_mod.QwenTTS(comp_mod.QwenTTSConfig())
    run_comp._load_model = lambda: None
    run_comp._register_voices = lambda: None
    run_comp._worker = lambda outputs: None

    class _InterruptQueue:
        def __init__(self):
            self.empty_calls = 0
            self.drained = False

        def empty(self):
            self.empty_calls += 1
            return self.empty_calls > 1

        def get_nowait(self):
            self.drained = True
            raise comp_mod.Empty

        def put(self, item):
            return None

    run_comp._task_queue = _InterruptQueue()
    monkeypatch.setattr(comp_mod.threading, "Thread", _SyncThread)
    run_comp.run(
        comp_mod.QwenTTSInputs(
            text=_FakeRecv([EOS.END, None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="stop"), None]),
        ),
        comp_mod.QwenTTSOutputs(
            audio=SimpleNamespace(send=lambda value: None),
            text=SimpleNamespace(send=lambda value: None),
        ),
    )
    assert run_comp._task_queue.drained is True


def test_qwen_tts_package_missing_attr() -> None:
    import src.core.conduit.qwen_tts as qwen_pkg

    with pytest.raises(AttributeError):
        _ = qwen_pkg.missing_attr
