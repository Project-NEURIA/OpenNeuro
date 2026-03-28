from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_qwen_tts_configs_and_processor(monkeypatch) -> None:
    import transformers.utils.generic as transformers_generic

    if not hasattr(transformers_generic, "check_model_inputs"):
        monkeypatch.setattr(
            transformers_generic,
            "check_model_inputs",
            lambda *args, **kwargs: lambda func: func,
            raising=False,
        )

    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts as cfg_mod
    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts_tokenizer_v2 as tok_cfg_mod
    import src.core.conduit.qwen_tts.tts_model.processing_qwen3_tts as proc_mod

    rope_calls = []
    layer_calls = []
    monkeypatch.setattr(
        cfg_mod, "rope_config_validation", lambda cfg: rope_calls.append(cfg)
    )
    monkeypatch.setattr(
        cfg_mod,
        "layer_type_validation",
        lambda layer_types: layer_calls.append(list(layer_types)),
    )

    speaker_cfg = cfg_mod.Qwen3TTSSpeakerEncoderConfig(sample_rate=16000, enc_dim=256)
    assert speaker_cfg.sample_rate == 16000
    assert speaker_cfg.enc_dim == 256

    code_cfg = cfg_mod.Qwen3TTSTalkerCodePredictorConfig(
        num_key_value_heads=None,
        rope_scaling={"type": "linear", "factor": 2.0},
        use_sliding_window=True,
        max_window_layers=1,
        num_hidden_layers=3,
    )
    assert code_cfg.num_key_value_heads == code_cfg.num_attention_heads
    assert code_cfg.rope_scaling["rope_type"] == "linear"
    assert code_cfg.layer_types == [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    assert rope_calls and layer_calls

    talker_default = cfg_mod.Qwen3TTSTalkerConfig(code_predictor_config=None)
    assert isinstance(
        talker_default.code_predictor_config,
        cfg_mod.Qwen3TTSTalkerCodePredictorConfig,
    )
    talker_existing = cfg_mod.Qwen3TTSTalkerConfig(code_predictor_config=code_cfg)
    assert talker_existing.code_predictor_config is code_cfg
    talker_dict = cfg_mod.Qwen3TTSTalkerConfig(
        code_predictor_config={"hidden_size": 128}
    )
    assert talker_dict.code_predictor_config.hidden_size == 128

    tts_cfg = cfg_mod.Qwen3TTSConfig(
        talker_config={"hidden_size": 64},
        speaker_encoder_config={"enc_dim": 32},
        tokenizer_type="demo",
        tts_model_type="base",
    )
    assert tts_cfg.talker_config.hidden_size == 64
    assert tts_cfg.speaker_encoder_config.enc_dim == 32
    assert tts_cfg.tokenizer_type == "demo"
    assert tts_cfg.tts_model_type == "base"

    decoder_cfg = tok_cfg_mod.Qwen3TTSTokenizerV2DecoderConfig(num_hidden_layers=3)
    assert decoder_cfg.layer_types == ["sliding_attention"] * 3
    full_tok_cfg = tok_cfg_mod.Qwen3TTSTokenizerV2Config(
        decoder_config={"num_hidden_layers": 2},
        input_sample_rate=22050,
        output_sample_rate=44100,
    )
    assert full_tok_cfg.decoder_config.num_hidden_layers == 2
    assert full_tok_cfg.input_sample_rate == 22050
    assert full_tok_cfg.output_sample_rate == 44100

    def _proc_init(self, tokenizer=None, chat_template=None):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    monkeypatch.setattr(proc_mod.ProcessorMixin, "__init__", _proc_init)
    monkeypatch.setattr(
        proc_mod.ProcessorMixin,
        "apply_chat_template",
        lambda self, conversations, chat_template=None, **kwargs: conversations,
    )
    monkeypatch.setattr(
        proc_mod.Qwen3TTSProcessor,
        "_merge_kwargs",
        lambda self, kwargs_cls, tokenizer_init_kwargs=None, **kwargs: {
            "text_kwargs": {"padding": True, "padding_side": "left"}
        },
    )

    class _Tokenizer:
        init_kwargs = {"padding": False}
        model_input_names = ["input_ids", "attention_mask", "input_ids"]

        def __init__(self):
            self.calls = []

        def __call__(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return {
                "input_ids": [[1, 2] for _ in text],
                "attention_mask": [[1, 1] for _ in text],
            }

        def batch_decode(self, *args, **kwargs):
            return ["batch-decoded"]

        def decode(self, *args, **kwargs):
            return "decoded"

    tokenizer = _Tokenizer()
    processor = proc_mod.Qwen3TTSProcessor(tokenizer=tokenizer, chat_template="chat")
    with pytest.raises(ValueError):
        processor()

    batch = processor("hello", return_tensors="pt")
    assert batch["input_ids"].tolist() == [[1, 2]]
    batch2 = processor(["a", "b"])
    assert batch2["attention_mask"] == [[1, 1], [1, 1]]
    assert processor.batch_decode([]) == ["batch-decoded"]
    assert processor.decode([]) == "decoded"
    assert processor.apply_chat_template([{"role": "user"}]) == [[{"role": "user"}]]
    assert processor.model_input_names == ["input_ids", "attention_mask"]


def test_qwen3_tts_tokenizer_paths(monkeypatch) -> None:
    import transformers.utils.generic as transformers_generic

    if not hasattr(transformers_generic, "check_model_inputs"):
        monkeypatch.setattr(
            transformers_generic,
            "check_model_inputs",
            lambda *args, **kwargs: lambda func: func,
            raising=False,
        )

    import src.core.conduit.qwen_tts.tts_model.qwen3_tts_tokenizer as tok_mod

    register_calls = []
    monkeypatch.setattr(
        tok_mod.AutoConfig,
        "register",
        lambda name, cls: register_calls.append(("config", name, cls)),
    )
    monkeypatch.setattr(
        tok_mod.AutoModel,
        "register",
        lambda cfg_cls, model_cls: register_calls.append(("model", cfg_cls, model_cls)),
    )

    class _Batch(dict):
        def to(self, *_args, **_kwargs):
            return self

    class _FeatureExtractor:
        sampling_rate = 24000

        def __call__(self, raw_audio, sampling_rate, return_tensors):
            max_len = max(len(x) for x in raw_audio)
            values = []
            masks = []
            for wav in raw_audio:
                pad = max_len - len(wav)
                values.append(np.pad(wav, (0, pad)))
                masks.append(np.concatenate([np.ones(len(wav)), np.zeros(pad)]))
            return _Batch(
                input_values=torch.tensor(values, dtype=torch.float32).unsqueeze(1),
                padding_mask=torch.tensor(masks, dtype=torch.float32).unsqueeze(1),
            )

    class _FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(model_type="qwen3_tts_tokenizer_12hz")
            self.dtype = torch.float32
            self.device = None
            self.encode_calls = []
            self.decode_calls = []

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def encode(self, input_values, padding_mask, return_dict=True):
            self.encode_calls.append(
                (input_values.shape, padding_mask.shape, return_dict)
            )
            return SimpleNamespace(
                audio_codes=[torch.tensor([[1, 2], [3, 4]], dtype=torch.long)]
            )

        def decode(self, audio_codes, *args, return_dict=True):
            self.decode_calls.append((audio_codes.shape, args, return_dict))
            batch = audio_codes.shape[0]
            return SimpleNamespace(
                audio_values=[
                    torch.arange(3, dtype=torch.float32) + i for i in range(batch)
                ]
            )

        def get_model_type(self):
            return self.config.model_type

        def get_input_sample_rate(self):
            return 24000

        def get_output_sample_rate(self):
            return 24000

        def get_encode_downsample_rate(self):
            return 1920

        def get_decode_upsample_rate(self):
            return 1920

    model = _FakeModel()
    monkeypatch.setattr(
        tok_mod.AutoFeatureExtractor,
        "from_pretrained",
        lambda path: _FeatureExtractor(),
    )
    monkeypatch.setattr(
        tok_mod.AutoModel,
        "from_pretrained",
        lambda path, **kwargs: model,
    )

    tokenizer = tok_mod.Qwen3TTSTokenizer.from_pretrained("demo", device_map="cpu")
    assert tokenizer.model is model
    assert tokenizer.device.type == "cpu"
    assert register_calls[0][0] == "config"
    assert register_calls[1][0] == "model"

    assert tokenizer._is_probably_base64("data:audio/wav;base64,AAAA") is True
    assert tokenizer._is_probably_base64("a" * 300) is True
    assert tokenizer._is_probably_base64("x/y.wav") is False
    assert tokenizer._is_url("https://example.com/a.wav") is True
    assert tokenizer._is_url("not-a-url") is False
    assert (
        tokenizer._decode_base64_to_wav_bytes(
            "data:audio/wav;base64," + base64.b64encode(b"abc").decode("ascii")
        )
        == b"abc"
    )

    class _URLResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"wav-bytes"

    monkeypatch.setattr(tok_mod.urllib.request, "urlopen", lambda url: _URLResp())
    sf_reads = iter(
        [
            (np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32), 48000),
            (np.array([1.0, 2.0], dtype=np.float32), 24000),
        ]
    )
    monkeypatch.setattr(tok_mod.sf, "read", lambda *args, **kwargs: next(sf_reads))
    monkeypatch.setattr(
        tok_mod.librosa,
        "load",
        lambda path, sr=None, mono=True: (
            np.array([1.0, 2.0], dtype=np.float32),
            22050,
        ),
    )
    monkeypatch.setattr(
        tok_mod.librosa, "resample", lambda y, orig_sr, target_sr: np.asarray(y) * 2
    )

    wav_url = tokenizer.load_audio("https://example.com/a.wav", target_sr=24000)
    wav_b64 = tokenizer.load_audio("data:audio/wav;base64,AAAA", target_sr=24000)
    wav_file = tokenizer.load_audio("demo.wav", target_sr=24000)
    assert wav_url.dtype == np.float32 and wav_url.ndim == 1
    assert wav_b64.dtype == np.float32
    assert wav_file.dtype == np.float32

    monkeypatch.setattr(
        tokenizer,
        "load_audio",
        lambda x, target_sr: np.array([1.0, 2.0], dtype=np.float32),
    )
    assert tokenizer._normalize_audio_inputs([], sr=None) == []
    assert len(tokenizer._normalize_audio_inputs("a.wav", sr=None)) == 1
    assert len(tokenizer._normalize_audio_inputs(["a.wav", "b.wav"], sr=None)) == 2
    with pytest.raises(ValueError):
        tokenizer._normalize_audio_inputs(np.array([1.0], dtype=np.float32), sr=None)
    with pytest.raises(TypeError):
        tokenizer._normalize_audio_inputs(
            [np.array([1.0], dtype=np.float32), "x"], sr=16000
        )

    waveforms = tokenizer._normalize_audio_inputs(
        [np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)],
        sr=16000,
    )
    assert waveforms[0].dtype == np.float32

    monkeypatch.setattr(
        tokenizer,
        "_normalize_audio_inputs",
        lambda audios, sr=None: [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3], dtype=np.float32),
        ],
    )
    encoded = tokenizer.encode("demo.wav")
    assert encoded.audio_codes[0].shape == (2, 2)
    assert model.encode_calls[0][0][0] == 2

    wavs, sr = tokenizer.decode(
        SimpleNamespace(audio_codes=[torch.tensor([[1, 2], [3, 4]], dtype=torch.long)])
    )
    assert len(wavs) == 1
    assert sr == 24000

    model.config.model_type = "qwen3_tts_tokenizer_25hz"
    with pytest.raises(ValueError):
        tokenizer.decode({"audio_codes": [torch.tensor([1, 2], dtype=torch.long)]})

    wavs2, _ = tokenizer.decode(
        {
            "audio_codes": [torch.tensor([1, 2], dtype=torch.long)],
            "xvectors": [torch.tensor([0.1, 0.2], dtype=torch.float32)],
            "ref_mels": [torch.tensor([[0.1, 0.2]], dtype=torch.float32)],
        }
    )
    assert len(wavs2) == 1

    wavs3, _ = tokenizer.decode(
        [
            {
                "audio_codes": torch.tensor([1, 2], dtype=torch.long),
                "xvectors": torch.tensor([0.1, 0.2], dtype=torch.float32),
                "ref_mels": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            }
        ]
    )
    assert len(wavs3) == 1

    model.config.model_type = "qwen3_tts_tokenizer_12hz"
    wavs4, _ = tokenizer.decode(
        {"audio_codes": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)}
    )
    assert len(wavs4) == 1

    with pytest.raises(TypeError):
        tokenizer.decode(123)

    model.config.model_type = "unknown"
    with pytest.raises(ValueError):
        tokenizer.decode({"audio_codes": [torch.tensor([1], dtype=torch.long)]})

    model.config.model_type = "qwen3_tts_tokenizer_12hz"
    assert tokenizer.get_model_type() == "qwen3_tts_tokenizer_12hz"
    assert tokenizer.get_input_sample_rate() == 24000
    assert tokenizer.get_output_sample_rate() == 24000
    assert tokenizer.get_encode_downsample_rate() == 1920
    assert tokenizer.get_decode_upsample_rate() == 1920


def test_qwen_tts_config_and_tokenizer_remaining_paths(monkeypatch) -> None:
    import transformers.utils.generic as transformers_generic

    if not hasattr(transformers_generic, "check_model_inputs"):
        monkeypatch.setattr(
            transformers_generic,
            "check_model_inputs",
            lambda *args, **kwargs: lambda func: func,
            raising=False,
        )

    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts as cfg_mod
    import src.core.conduit.qwen_tts.tts_model.qwen3_tts_tokenizer as tok_mod

    talker_cfg = cfg_mod.Qwen3TTSTalkerConfig(rope_scaling={"type": "default"})
    assert talker_cfg.rope_scaling["rope_type"] == "default"

    monkeypatch.setattr(tok_mod.AutoConfig, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(tok_mod.AutoModel, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tok_mod.AutoFeatureExtractor, "from_pretrained", lambda path: object()
    )

    class _NoParamModel:
        def __init__(self):
            self.config = SimpleNamespace(model_type="qwen3_tts_tokenizer_12hz")
            self.device = None

        def parameters(self):
            return iter(())

        def decode(self, audio_codes, *args, return_dict=True):
            return SimpleNamespace(
                audio_values=[torch.arange(audio_codes.shape[0], dtype=torch.float32)]
            )

        def get_model_type(self):
            return self.config.model_type

        def get_input_sample_rate(self):
            return 24000

        def get_output_sample_rate(self):
            return 24000

        def get_encode_downsample_rate(self):
            return 1920

        def get_decode_upsample_rate(self):
            return 1920

    monkeypatch.setattr(
        tok_mod.AutoModel, "from_pretrained", lambda path, **kwargs: _NoParamModel()
    )
    tokenizer = tok_mod.Qwen3TTSTokenizer.from_pretrained("demo")
    assert tokenizer.device.type == "cpu"
    decoded, sample_rate = tokenizer.decode(
        {"audio_codes": torch.tensor([1, 2], dtype=torch.long)}
    )
    assert len(decoded) == 1
    assert sample_rate == 24000
