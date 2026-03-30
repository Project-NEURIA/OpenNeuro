from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _ensure_transformers_patch(monkeypatch) -> None:
    import transformers.utils.generic as transformers_generic

    monkeypatch.setattr(
        transformers_generic,
        "check_model_inputs",
        lambda *args, **kwargs: lambda func: func,
        raising=False,
    )


def _tiny_qwen_config(cfg_mod):
    return cfg_mod.Qwen3TTSConfig(
        talker_config={
            "vocab_size": 32,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_code_groups": 3,
            "text_hidden_size": 4,
            "max_position_embeddings": 16,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1],
                "interleaved": False,
            },
            "code_predictor_config": {
                "vocab_size": 16,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "num_code_groups": 3,
                "max_position_embeddings": 16,
                "pad_token_id": 0,
            },
            "spk_id": {"alice": [1], "dialectspeaker": [2]},
            "spk_is_dialect": {"alice": False, "dialectspeaker": "mandarin"},
            "codec_language_id": {"english": 7, "mandarin": 8},
            "pad_token_id": 0,
            "text_vocab_size": 64,
            "codec_eos_token_id": 18,
            "codec_think_id": 19,
            "codec_nothink_id": 20,
            "codec_think_bos_id": 21,
            "codec_think_eos_id": 22,
            "codec_pad_id": 23,
            "codec_bos_id": 24,
        },
        speaker_encoder_config={
            "mel_dim": 4,
            "enc_dim": 4,
            "enc_channels": [4, 4, 4, 4, 12],
            "enc_kernel_sizes": [3, 3, 3, 3, 1],
            "enc_dilations": [1, 1, 1, 1, 1],
            "enc_attention_channels": 2,
            "enc_res2net_scale": 2,
            "enc_se_channels": 2,
            "sample_rate": 24000,
        },
        tokenizer_type="tok",
        tts_model_size="tiny",
        tts_model_type="base",
        im_start_token_id=1,
        im_end_token_id=2,
        tts_pad_token_id=3,
        tts_bos_token_id=4,
        tts_eos_token_id=5,
    )


def test_qwen_tts_modeling_low_level_paths(monkeypatch) -> None:
    _ensure_transformers_patch(monkeypatch)

    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts as cfg_mod
    import src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts as mod

    monkeypatch.setitem(
        mod.ROPE_INIT_FUNCTIONS,
        "default",
        lambda config, device=None: (
            torch.ones(
                getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
                // 2,
                dtype=torch.float32,
            ),
            1.0,
        ),
    )

    snapshot_calls = []
    monkeypatch.setattr(mod.huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(
        mod,
        "snapshot_download",
        lambda *args, **kwargs: snapshot_calls.append((args, kwargs)) or "folder",
    )
    assert (
        mod.download_weights_from_hf_specific(
            "demo", cache_dir="cache", allow_patterns=["a", "b"], revision="main"
        )
        == "folder"
    )
    assert len(snapshot_calls) == 2

    res2 = mod.Res2NetBlock(4, 4, scale=2, kernel_size=3, dilation=1)
    assert res2(torch.ones((1, 4, 5))).shape == (1, 4, 5)
    se = mod.SqueezeExcitationBlock(4, 2, 4)
    assert se(torch.ones((1, 4, 5))).shape == (1, 4, 5)
    pool = mod.AttentiveStatisticsPooling(4, attention_channels=2)
    mask = pool._length_to_mask(torch.tensor([2, 3]), dtype=torch.float32)
    assert mask.shape == (2, 3)
    mean, std = pool._compute_statistics(
        torch.ones((1, 4, 3)), torch.ones((1, 4, 3)) / 3
    )
    assert mean.shape == std.shape == (1, 4)
    assert pool(torch.ones((1, 4, 4))).shape == (1, 8, 1)
    tdnn = mod.TimeDelayNetBlock(4, 4, kernel_size=3, dilation=1)
    assert tdnn(torch.ones((1, 4, 5))).shape == (1, 4, 5)
    se_res2 = mod.SqueezeExcitationRes2NetBlock(
        4, 4, res2net_scale=2, se_channels=2, kernel_size=3, dilation=1
    )
    assert se_res2(torch.ones((1, 4, 5))).shape == (1, 4, 5)

    bad_cfg = cfg_mod.Qwen3TTSSpeakerEncoderConfig(
        mel_dim=4,
        enc_dim=4,
        enc_channels=[4, 4],
        enc_kernel_sizes=[3],
        enc_dilations=[1, 1],
    )
    with pytest.raises(ValueError):
        mod.Qwen3TTSSpeakerEncoder(bad_cfg)

    qcfg = _tiny_qwen_config(cfg_mod)
    speaker = mod.Qwen3TTSSpeakerEncoder(qcfg.speaker_encoder_config)
    assert speaker(torch.ones((1, 6, 4))).shape == (1, 4)

    assert torch.allclose(
        mod.dynamic_range_compression_torch(torch.tensor([1.0], dtype=torch.float32)),
        torch.tensor([0.0], dtype=torch.float32),
    )
    monkeypatch.setattr(
        mod,
        "librosa_mel_fn",
        lambda sr, n_fft, n_mels, fmin, fmax: np.ones(
            (n_mels, n_fft // 2 + 1), dtype=np.float32
        ),
    )
    mel = mod.mel_spectrogram(
        torch.tensor([[-1.5, 0.0, 1.5, 0.0]], dtype=torch.float32),
        n_fft=4,
        num_mels=4,
        sampling_rate=24000,
        hop_size=1,
        win_size=4,
        fmin=0,
        fmax=12000,
    )
    assert mel.shape[1] == 4

    base_pre = object.__new__(mod.Qwen3TTSPreTrainedModel)
    base_pre.config = SimpleNamespace(initializer_range=0.01)
    linear = torch.nn.Linear(2, 2)
    embedding = torch.nn.Embedding(4, 2, padding_idx=0)
    layer_norm = torch.nn.LayerNorm(2)
    base_pre._init_weights(linear)
    base_pre._init_weights(embedding)
    base_pre._init_weights(layer_norm)

    text_pre = object.__new__(mod.Qwen3TTSTalkerTextPreTrainedModel)
    text_pre.config = SimpleNamespace(initializer_range=0.01)
    text_pre._init_weights(torch.nn.Linear(2, 2))
    text_pre._init_weights(torch.nn.Embedding(4, 2, padding_idx=0))
    text_pre._init_weights(mod.Qwen3TTSRMSNorm(2))

    talker_rot = mod.Qwen3TTSTalkerRotaryEmbedding(qcfg.talker_config)
    talker_pos = torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long)
    cos, sin = talker_rot(torch.ones((1, 2, 4)), talker_pos)
    assert cos.shape == sin.shape

    tts_rot = mod.Qwen3TTSRotaryEmbedding(qcfg.talker_config.code_predictor_config)
    cos2, sin2 = tts_rot(
        torch.ones((1, 2, 4)), torch.tensor([[0, 1]], dtype=torch.long)
    )
    assert cos2.shape == sin2.shape == (1, 2, 4)

    rms = mod.Qwen3TTSRMSNorm(4)
    assert rms(torch.ones((1, 2, 4))).shape == (1, 2, 4)
    assert "eps=" in rms.extra_repr()
    assert torch.equal(
        mod.rotate_half(torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
        torch.tensor([[-3.0, -4.0, 1.0, 2.0]]),
    )
    kv = torch.ones((1, 1, 2, 4))
    assert mod.repeat_kv(kv, 2).shape == (1, 2, 2, 4)
    attn_out, attn_weights = mod.eager_attention_forward(
        SimpleNamespace(num_key_value_groups=1, training=False),
        torch.ones((1, 1, 2, 4)),
        torch.ones((1, 1, 2, 4)),
        torch.ones((1, 1, 2, 4)),
        torch.zeros((1, 1, 2, 2)),
        scaling=0.5,
    )
    assert attn_out.shape == (1, 2, 1, 4)
    assert attn_weights.shape == (1, 1, 2, 2)
    q = torch.ones((1, 1, 2, 4))
    k = torch.ones((1, 1, 2, 4))
    mm_cos = torch.ones((3, 1, 2, 4))
    mm_sin = torch.zeros((3, 1, 2, 4))
    q_mm, k_mm = mod.apply_multimodal_rotary_pos_emb(
        q, k, mm_cos, mm_sin, [1, 1], mrope_interleaved=False
    )
    assert q_mm.shape == k_mm.shape == q.shape
    q_mm2, k_mm2 = mod.apply_multimodal_rotary_pos_emb(
        q, k, mm_cos, mm_sin, [1, 2], mrope_interleaved=True
    )
    assert q_mm2.shape == k_mm2.shape == q.shape
    q_std, k_std = mod.apply_rotary_pos_emb(
        q, k, torch.ones((1, 2, 4)), torch.zeros((1, 2, 4))
    )
    assert q_std.shape == k_std.shape == q.shape


def test_qwen_tts_modeling_submodels(monkeypatch) -> None:
    _ensure_transformers_patch(monkeypatch)

    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts as cfg_mod
    import src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts as mod

    monkeypatch.setitem(
        mod.ROPE_INIT_FUNCTIONS,
        "default",
        lambda config, device=None: (
            torch.ones(
                getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
                // 2,
                dtype=torch.float32,
            ),
            1.0,
        ),
    )

    qcfg = _tiny_qwen_config(cfg_mod)
    code_cfg = qcfg.talker_config.code_predictor_config
    code_cfg._attn_implementation = "eager"
    talker_cfg = qcfg.talker_config
    talker_cfg._attn_implementation = "eager"

    position_embeddings = mod.Qwen3TTSRotaryEmbedding(code_cfg)(
        torch.ones((1, 2, 4)), torch.tensor([[0, 1]], dtype=torch.long)
    )
    attention = mod.Qwen3TTSAttention(code_cfg, layer_idx=0)

    class _Cache:
        def update(self, key_states, value_states, layer_idx, cache_kwargs):
            return key_states, value_states

    out, weights = attention(
        torch.ones((1, 2, 4)),
        position_embeddings,
        torch.zeros((1, 1, 2, 2)),
        past_key_values=_Cache(),
        cache_position=torch.tensor([0, 1]),
    )
    assert out.shape == (1, 2, 4)
    assert weights is not None

    code_cfg_alt = cfg_mod.Qwen3TTSTalkerCodePredictorConfig(**code_cfg.to_dict())
    code_cfg_alt._attn_implementation = "custom"
    monkeypatch.setitem(
        mod.ALL_ATTENTION_FUNCTIONS,
        "custom",
        lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs: (
            query.transpose(1, 2).contiguous(),
            torch.zeros((query.shape[0], 1, query.shape[2], key.shape[2])),
        ),
    )
    out_alt, _ = mod.Qwen3TTSAttention(code_cfg_alt, layer_idx=0)(
        torch.ones((1, 2, 4)),
        position_embeddings,
        torch.zeros((1, 1, 2, 2)),
    )
    assert out_alt.shape == (1, 2, 4)

    talker_position_embeddings = mod.Qwen3TTSTalkerRotaryEmbedding(talker_cfg)(
        torch.ones((1, 2, 4)),
        torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long),
    )
    talker_attention = mod.Qwen3TTSTalkerAttention(talker_cfg, layer_idx=0)
    talker_out, talker_weights = talker_attention(
        torch.ones((1, 2, 4)),
        talker_position_embeddings,
        torch.zeros((1, 1, 2, 2)),
        past_key_values=_Cache(),
        cache_position=torch.tensor([0, 1]),
    )
    assert talker_out.shape == (1, 2, 4)
    assert talker_weights is not None

    resize = mod.Qwen3TTSTalkerResizeMLP(4, 4, 4, "relu", bias=True)
    assert resize(torch.ones((1, 2, 4))).shape == (1, 2, 4)
    text_mlp = mod.Qwen3TTSTalkerTextMLP(code_cfg, intermediate_size=8)
    assert text_mlp(torch.ones((1, 2, 4))).shape == (1, 2, 4)

    decoder_layer = mod.Qwen3TTSDecoderLayer(code_cfg, layer_idx=0)
    assert decoder_layer(
        torch.ones((1, 2, 4)),
        attention_mask=torch.zeros((1, 1, 2, 2)),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        cache_position=torch.tensor([0, 1]),
        position_embeddings=position_embeddings,
    )[0].shape == (1, 2, 4)

    predictor_model = mod.Qwen3TTSTalkerCodePredictorModel(code_cfg, embedding_dim=4)
    assert predictor_model.get_input_embeddings() is predictor_model.codec_embedding
    predictor_model.set_input_embeddings("x")
    assert predictor_model.embed_tokens == "x"
    with pytest.raises(ValueError):
        predictor_model(input_ids=torch.ones((1, 1), dtype=torch.long))
    with pytest.raises(ValueError):
        predictor_model(inputs_embeds=torch.ones((1, 2, 4)), past_key_values="bad")
    warnings = []
    monkeypatch.setattr(
        mod.logger, "warning_once", lambda message: warnings.append(message)
    )
    predictor_model.gradient_checkpointing = True
    predictor_model.train()
    predictor_out = predictor_model(inputs_embeds=torch.ones((1, 2, 4)), use_cache=True)
    assert predictor_out.last_hidden_state.shape == (1, 2, 4)
    assert warnings

    predictor_gen = mod.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        code_cfg, talker_cfg
    )
    assert (
        predictor_gen.get_input_embeddings()
        is predictor_gen.model.get_input_embeddings()
    )
    assert predictor_gen.get_output_embeddings() is predictor_gen.lm_head
    predictor_input = "input"
    predictor_output = torch.nn.ModuleList(
        [
            torch.nn.Linear(4, 16, bias=False)
            for _ in range(code_cfg.num_code_groups - 1)
        ]
    )
    predictor_decoder = mod.Qwen3TTSTalkerCodePredictorModel(code_cfg, embedding_dim=4)
    predictor_gen.set_input_embeddings(predictor_input)
    assert predictor_gen.model.embed_tokens == predictor_input
    predictor_gen.set_output_embeddings(predictor_output)
    assert predictor_gen.lm_head is predictor_output
    predictor_gen.set_decoder(predictor_decoder)
    assert predictor_gen.model is predictor_decoder

    predictor_gen = mod.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        code_cfg, talker_cfg
    )
    predictor_gen.loss_function = lambda **kwargs: torch.tensor(1.0)
    ft_out = predictor_gen.forward_finetune(
        inputs_embeds=torch.ones((1, 3, 4)),
        labels=torch.tensor([[1, 2]], dtype=torch.long),
        use_cache=False,
    )
    assert ft_out.loss.item() == 1.0
    prefill = predictor_gen(inputs_embeds=torch.ones((1, 3, 4)), use_cache=False)
    assert prefill.logits.shape == (1, 3, 16)
    gen_stage = predictor_gen(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        generation_steps=1,
        use_cache=False,
    )
    assert gen_stage.logits.shape == (1, 1, 16)
    updated = predictor_gen._update_model_kwargs_for_generation(
        mod.Qwen3TTSTalkerCodePredictorOutputWithPast(past_key_values="pkv"), {"a": 1}
    )
    assert updated["past_key_values"] == "pkv"

    talker_decoder = mod.Qwen3TTSTalkerDecoderLayer(talker_cfg, layer_idx=0)
    talker_layer_out = talker_decoder(
        torch.ones((1, 2, 4)),
        attention_mask=torch.zeros((1, 1, 2, 2)),
        position_ids=torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long),
        cache_position=torch.tensor([0, 1]),
        position_embeddings=talker_position_embeddings,
    )
    assert talker_layer_out[0].shape == (1, 2, 4)

    talker_model = mod.Qwen3TTSTalkerModel(talker_cfg)
    assert talker_model.get_input_embeddings() is talker_model.codec_embedding
    assert talker_model.get_text_embeddings() is talker_model.text_embedding
    talker_model.set_input_embeddings("embed")
    assert talker_model.embed_tokens == "embed"
    with pytest.raises(ValueError):
        talker_model(
            input_ids=torch.ones((1, 1), dtype=torch.long),
            inputs_embeds=torch.ones((1, 1, 4)),
        )
    talker_model.gradient_checkpointing = True
    talker_model.train()
    warnings.clear()
    talker_model_out = talker_model(inputs_embeds=torch.ones((1, 2, 4)), use_cache=True)
    assert talker_model_out.last_hidden_state.shape == (1, 2, 4)

    talker_gen = mod.Qwen3TTSTalkerForConditionalGeneration(talker_cfg)
    talker_gen.loss_function = lambda **kwargs: torch.tensor(2.0)
    assert talker_gen.get_input_embeddings() is talker_gen.model.get_input_embeddings()
    assert talker_gen.get_text_embeddings() is talker_gen.model.get_text_embeddings()
    with pytest.raises(AttributeError):
        talker_gen.get_output_embeddings()
    talker_input = "codec"
    talker_output = torch.nn.Linear(4, 32, bias=False)
    talker_decoder = mod.Qwen3TTSTalkerModel(talker_cfg)
    talker_gen.set_input_embeddings(talker_input)
    assert talker_gen.model.embed_tokens == talker_input
    talker_gen.set_output_embeddings(talker_output)
    assert talker_gen.lm_head is talker_output
    assert talker_gen.codec_head is not talker_output
    talker_gen.set_decoder(talker_decoder)
    assert talker_gen.model is talker_decoder

    talker_gen = mod.Qwen3TTSTalkerForConditionalGeneration(talker_cfg)
    talker_gen.loss_function = lambda **kwargs: torch.tensor(2.0)
    talker_gen.code_predictor.forward_finetune = lambda inputs_embeds, labels: (
        SimpleNamespace(
            logits=torch.ones((1, 2, 16), dtype=torch.float32), loss=torch.tensor(3.0)
        )
    )
    logits, loss = talker_gen.forward_sub_talker_finetune(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.ones((1, 4), dtype=torch.float32),
    )
    assert logits.shape == (1, 2, 16)
    assert loss.item() == 3.0

    prefill_out = talker_gen(
        inputs_embeds=torch.ones((1, 3, 4)),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        trailing_text_hidden=torch.ones((1, 2, 4)),
        tts_pad_embed=torch.ones((1, 1, 4)),
        labels=torch.tensor([[1, 2, 3]], dtype=torch.long),
        use_cache=False,
    )
    assert prefill_out.loss.item() == 2.0
    assert prefill_out.logits.shape == (1, 3, 32)

    talker_gen.code_predictor.generate = lambda **kwargs: SimpleNamespace(
        sequences=torch.tensor([[2, 3]], dtype=torch.long)
    )
    gen_out = talker_gen(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        past_hidden=torch.ones((1, 1, 4)),
        trailing_text_hidden=torch.ones((1, 2, 4)),
        tts_pad_embed=torch.ones((1, 1, 4)),
        generation_step=0,
        use_cache=False,
    )
    assert gen_out.logits.shape == (1, 1, 32)
    gen_out2 = talker_gen(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        past_hidden=torch.ones((1, 1, 4)),
        trailing_text_hidden=torch.ones((1, 1, 4)),
        tts_pad_embed=torch.ones((1, 1, 4)),
        generation_step=2,
        use_cache=False,
    )
    assert gen_out2.logits.shape == (1, 1, 32)
    rope_ids, rope_delta = talker_gen.get_rope_index(
        attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.long)
    )
    assert rope_ids.shape == (3, 1, 3)
    assert rope_delta.shape == (1, 1)
    updated = talker_gen._update_model_kwargs_for_generation(
        mod.Qwen3TTSTalkerOutputWithPast(
            past_key_values="pkv",
            past_hidden="past_hidden",
            generation_step=4,
            trailing_text_hidden="trail",
            tts_pad_embed="pad",
        ),
        {"x": 1},
    )
    assert updated["past_hidden"] == "past_hidden"
    assert updated["tts_pad_embed"] == "pad"


def test_qwen_tts_modeling_full_model_paths(monkeypatch, tmp_path: Path) -> None:
    _ensure_transformers_patch(monkeypatch)

    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts as cfg_mod
    import src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts as mod

    monkeypatch.setitem(
        mod.ROPE_INIT_FUNCTIONS,
        "default",
        lambda config, device=None: (
            torch.ones(
                getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
                // 2,
                dtype=torch.float32,
            ),
            1.0,
        ),
    )

    qcfg = _tiny_qwen_config(cfg_mod)
    model = mod.Qwen3TTSForConditionalGeneration(qcfg)
    model.load_speech_tokenizer("speech")
    model.load_generate_config({"top_k": 3})
    assert model.get_supported_speakers() == qcfg.talker_config.spk_id.keys()
    assert model.get_supported_languages() == model.supported_languages

    root = tmp_path / "qwen"
    (root / "speech_tokenizer").mkdir(parents=True)
    (root / "speech_tokenizer" / "config.json").write_text("{}", encoding="utf-8")
    gen_path = root / "generation_config.json"
    gen_path.write_text(json.dumps({"temperature": 0.5}), encoding="utf-8")
    base_instance = mod.Qwen3TTSForConditionalGeneration(qcfg)
    monkeypatch.setattr(
        mod.Qwen3TTSPreTrainedModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: base_instance),
    )
    monkeypatch.setattr(
        mod, "download_weights_from_hf_specific", lambda *args, **kwargs: str(root)
    )
    monkeypatch.setattr(
        mod,
        "cached_file",
        lambda model_name, filename, **kwargs: str(root / filename),
    )
    monkeypatch.setattr(
        mod.Qwen3TTSTokenizer,
        "from_pretrained",
        classmethod(lambda cls, path, *args, **kwargs: SimpleNamespace(path=path)),
    )
    loaded = mod.Qwen3TTSForConditionalGeneration.from_pretrained(
        "remote-model", config=qcfg
    )
    assert loaded.speech_tokenizer.path.endswith("speech_tokenizer")
    assert loaded.generate_config["temperature"] == 0.5

    monkeypatch.setattr(
        mod,
        "mel_spectrogram",
        lambda audio, **kwargs: torch.ones((1, 4, 4), dtype=torch.float32),
    )
    monkeypatch.setattr(
        model.speaker_encoder,
        "forward",
        lambda hidden_states: torch.ones((1, 4), dtype=torch.float32),
    )
    speaker_emb = model.extract_speaker_embedding(np.ones(8, dtype=np.float32), 24000)
    assert speaker_emb.shape == (4,)

    voice_clone_prompt = {
        "ref_code": [
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[1, 2, 3]], dtype=torch.long),
        ],
        "ref_spk_embedding": [
            torch.ones((1, 4), dtype=torch.float32),
            torch.ones((1, 4), dtype=torch.float32) * 2,
        ],
        "x_vector_only_mode": [False, True],
        "icl_mode": [True, False],
    }
    speaker_prompts = model.generate_speaker_prompt(voice_clone_prompt)
    assert len(speaker_prompts) == 2
    icl_input, trailing = model.generate_icl_prompt(
        text_id=torch.tensor([[4, 5, 6, 7, 8, 9]], dtype=torch.long),
        ref_id=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        ref_code=torch.tensor([[1, 2, 3]], dtype=torch.long),
        tts_pad_embed=torch.ones((1, 1, 4), dtype=torch.float32),
        tts_eos_embed=torch.ones((1, 1, 4), dtype=torch.float32),
        non_streaming_mode=False,
    )
    assert icl_input.shape[0] == trailing.shape[0] == 1

    model.speech_tokenizer = SimpleNamespace(
        model=SimpleNamespace(decoder=lambda x: x.float()),
        get_output_sample_rate=lambda: 24000,
    )

    def _fake_talker_generate(**kwargs):
        return SimpleNamespace(
            hidden_states=[
                (
                    (torch.ones((2, 2, 4), dtype=torch.float32),),
                    torch.tensor([[1, 2, 18], [18, 2, 3]], dtype=torch.long),
                ),
                (
                    (torch.full((2, 2, 4), 2.0, dtype=torch.float32),),
                    torch.tensor([[4, 18, 1], [5, 6, 7]], dtype=torch.long),
                ),
            ]
        )

    model.talker.generate = _fake_talker_generate
    base_result = model.generate(
        input_ids=[
            torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long),
            torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long),
        ],
        languages=["english", "auto"],
        speakers=["alice", "dialectspeaker"],
        non_streaming_mode=False,
    )
    assert len(base_result[0]) == len(base_result[1]) == 2

    clone_result = model.generate(
        input_ids=[
            torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long),
            torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long),
        ],
        ref_ids=[
            torch.tensor([[1, 2, 3, 24, 25]], dtype=torch.long),
            torch.tensor([[1, 2, 3, 24, 25]], dtype=torch.long),
        ],
        voice_clone_prompt=voice_clone_prompt,
        languages=["auto", "english"],
        speakers=[None, None],
        non_streaming_mode=True,
    )
    assert len(clone_result[0]) == len(clone_result[1]) == 2


def test_qwen_tts_modeling_remaining_paths(monkeypatch) -> None:
    _ensure_transformers_patch(monkeypatch)

    import src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts as cfg_mod
    import src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts as mod

    monkeypatch.setitem(
        mod.ROPE_INIT_FUNCTIONS,
        "default",
        lambda config, device=None: (
            torch.ones(
                getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
                // 2,
                dtype=torch.float32,
            ),
            1.0,
        ),
    )

    # Res2Net branch with scale > 2.
    res2 = mod.Res2NetBlock(6, 6, scale=3, kernel_size=3, dilation=1)
    assert res2(torch.ones((1, 6, 5))).shape == (1, 6, 5)

    code_no_rope = cfg_mod.Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        num_code_groups=3,
        max_position_embeddings=8,
        pad_token_id=0,
        rope_scaling=None,
    )
    code_no_rope._attn_implementation = "eager"
    code_no_rope.rope_scaling = None
    talker_no_rope = cfg_mod.Qwen3TTSTalkerConfig(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_code_groups=3,
        text_hidden_size=4,
        max_position_embeddings=8,
        text_vocab_size=64,
        pad_token_id=0,
        rope_scaling=None,
        code_predictor_config=code_no_rope,
    )
    talker_no_rope._attn_implementation = "eager"
    assert mod.Qwen3TTSRotaryEmbedding(code_no_rope).rope_type == "default"
    assert mod.Qwen3TTSTalkerRotaryEmbedding(talker_no_rope).rope_type == "default"

    talker_custom = cfg_mod.Qwen3TTSTalkerConfig(
        **{
            **talker_no_rope.to_dict(),
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1],
                "interleaved": False,
            },
        }
    )
    talker_custom._attn_implementation = "custom"
    monkeypatch.setitem(
        mod.ALL_ATTENTION_FUNCTIONS,
        "custom",
        lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs: (
            query.transpose(1, 2).contiguous(),
            torch.zeros(
                (
                    query.shape[0],
                    module.num_key_value_groups,
                    query.shape[2],
                    key.shape[2],
                )
            ),
        ),
    )
    talker_pos = mod.Qwen3TTSTalkerRotaryEmbedding(talker_custom)(
        torch.ones((1, 2, 4)),
        torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long),
    )
    custom_out, _ = mod.Qwen3TTSTalkerAttention(talker_custom, layer_idx=0)(
        torch.ones((1, 2, 4)),
        talker_pos,
        torch.zeros((1, 1, 2, 2)),
    )
    assert custom_out.shape == (1, 2, 4)

    decoder_layer = mod.Qwen3TTSDecoderLayer(code_no_rope, layer_idx=0)
    dec_out = decoder_layer(
        torch.ones((1, 2, 4)),
        attention_mask=torch.zeros((1, 1, 2, 2)),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        cache_position=torch.tensor([0, 1]),
        position_embeddings=mod.Qwen3TTSRotaryEmbedding(code_no_rope)(
            torch.ones((1, 2, 4)), torch.tensor([[0, 1]], dtype=torch.long)
        ),
        output_attentions=True,
    )
    assert dec_out[1] is not None

    sliding_code = cfg_mod.Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        num_code_groups=3,
        max_position_embeddings=8,
        pad_token_id=0,
        use_sliding_window=True,
        max_window_layers=1,
    )
    sliding_code._attn_implementation = "eager"
    predictor_model = mod.Qwen3TTSTalkerCodePredictorModel(
        sliding_code, embedding_dim=4
    )
    with pytest.raises(ValueError):
        predictor_model()
    predictor_model.eval()
    predictor_out = predictor_model(
        inputs_embeds=torch.ones((1, 2, 4)),
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
    )
    assert predictor_out.past_key_values is not None
    assert predictor_out.hidden_states is not None
    assert predictor_out.attentions is not None

    talker_cfg = cfg_mod.Qwen3TTSTalkerConfig(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_code_groups=3,
        text_hidden_size=4,
        max_position_embeddings=8,
        text_vocab_size=64,
        pad_token_id=0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [1, 1],
            "interleaved": False,
        },
        code_predictor_config=sliding_code,
    )
    talker_cfg._attn_implementation = "eager"
    predictor_gen_proj = mod.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        cfg_mod.Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            num_code_groups=3,
            max_position_embeddings=8,
            pad_token_id=0,
        ),
        talker_cfg,
    )
    assert isinstance(predictor_gen_proj.small_to_mtp_projection, torch.nn.Linear)
    assert predictor_gen_proj.get_decoder() is predictor_gen_proj.model

    predictor_gen = mod.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        sliding_code, talker_cfg
    )
    predictor_gen.loss_function = lambda **kwargs: torch.tensor(4.0)
    out = predictor_gen(
        inputs_embeds=torch.ones((1, 3, 4)),
        labels=torch.tensor([[1, 2, 3]], dtype=torch.long),
        use_cache=False,
    )
    assert out.loss.item() == 4.0

    talker_decoder = mod.Qwen3TTSTalkerDecoderLayer(talker_cfg, layer_idx=0)
    talker_layer = talker_decoder(
        torch.ones((1, 2, 4)),
        attention_mask=torch.zeros((1, 1, 2, 2)),
        position_ids=torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long),
        cache_position=torch.tensor([0, 1]),
        position_embeddings=mod.Qwen3TTSTalkerRotaryEmbedding(talker_cfg)(
            torch.ones((1, 2, 4)),
            torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long),
        ),
        output_attentions=True,
    )
    assert talker_layer[1] is not None

    talker_model = mod.Qwen3TTSTalkerModel(talker_cfg)
    talker_model.set_input_embeddings(
        torch.nn.Embedding(talker_cfg.vocab_size, talker_cfg.hidden_size)
    )
    talker_forward = talker_model(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
    )
    assert talker_forward.past_key_values is not None
    assert talker_forward.hidden_states is not None
    assert talker_forward.attentions is not None
    talker_forward_4d = talker_model(
        inputs_embeds=torch.ones((1, 2, 4)),
        position_ids=torch.tensor(
            [[[9, 9]], [[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long
        ),
        use_cache=False,
    )
    assert talker_forward_4d.last_hidden_state.shape == (1, 2, 4)

    talker_gen = mod.Qwen3TTSTalkerForConditionalGeneration(talker_cfg)
    assert talker_gen.get_decoder() is talker_gen.model
    talker_gen.code_predictor.generate = lambda **kwargs: SimpleNamespace(
        sequences=torch.tensor([[2, 3]], dtype=torch.long)
    )
    talker_gen.rope_deltas = torch.tensor([[1]], dtype=torch.long)
    talker_gen_out = talker_gen(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        past_hidden=torch.ones((1, 1, 4), dtype=torch.float32),
        trailing_text_hidden=torch.ones((1, 1, 4), dtype=torch.float32),
        tts_pad_embed=torch.ones((1, 1, 4), dtype=torch.float32),
        generation_step=2,
        cache_position=torch.tensor([1], dtype=torch.long),
        use_cache=False,
    )
    assert talker_gen_out.logits.shape == (1, 1, 32)

    qcfg_non_base = _tiny_qwen_config(cfg_mod)
    qcfg_non_base.tts_model_type = "chat"
    assert mod.Qwen3TTSForConditionalGeneration(qcfg_non_base).speaker_encoder is None

    qcfg = _tiny_qwen_config(cfg_mod)
    missing_tok_model = mod.Qwen3TTSForConditionalGeneration(qcfg)
    monkeypatch.setattr(
        mod.Qwen3TTSPreTrainedModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: missing_tok_model),
    )
    monkeypatch.setattr(
        mod, "download_weights_from_hf_specific", lambda *args, **kwargs: "folder"
    )
    monkeypatch.setattr(mod, "cached_file", lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        mod.Qwen3TTSForConditionalGeneration.from_pretrained(
            "broken-model", config=qcfg
        )

    icl_model = mod.Qwen3TTSForConditionalGeneration(qcfg)
    icl_input, trailing = icl_model.generate_icl_prompt(
        text_id=torch.tensor([[4]], dtype=torch.long),
        ref_id=torch.empty((1, 0), dtype=torch.long),
        ref_code=torch.tensor([[1, 2, 3]], dtype=torch.long),
        tts_pad_embed=torch.ones((1, 1, 4), dtype=torch.float32),
        tts_eos_embed=torch.ones((1, 1, 4), dtype=torch.float32),
        non_streaming_mode=False,
    )
    assert icl_input.shape[1] == 2
    assert trailing.shape[-1] == 4

    gen_model = mod.Qwen3TTSForConditionalGeneration(qcfg)
    gen_model.speech_tokenizer = SimpleNamespace(
        model=SimpleNamespace(decoder=lambda x: x.float()),
        get_output_sample_rate=lambda: 24000,
    )

    def _fake_talker_generate(**kwargs):
        batch = kwargs["inputs_embeds"].shape[0]
        return SimpleNamespace(
            hidden_states=[
                (
                    (torch.ones((batch, 2, 4), dtype=torch.float32),),
                    torch.tensor([[1, 2, 18]] * batch, dtype=torch.long),
                ),
            ]
        )

    gen_model.talker.generate = _fake_talker_generate
    instruct_result = gen_model.generate(
        input_ids=[torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long)],
        instruct_ids=[torch.tensor([[9, 10]], dtype=torch.long)],
        languages=["auto"],
        speakers=None,
        non_streaming_mode=False,
    )
    assert len(instruct_result[0]) == 1

    voice_clone_prompt = {
        "ref_code": [torch.tensor([[1, 2, 3]], dtype=torch.long)],
        "ref_spk_embedding": [torch.ones((1, 4), dtype=torch.float32)],
        "x_vector_only_mode": [False],
        "icl_mode": [False],
    }
    clone_none_speaker = gen_model.generate(
        input_ids=[torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long)],
        voice_clone_prompt=voice_clone_prompt,
        languages=["auto"],
        speakers=[None],
        non_streaming_mode=False,
    )
    assert len(clone_none_speaker[0]) == 1

    with pytest.raises(NotImplementedError):
        gen_model.generate(
            input_ids=[
                torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long)
            ],
            languages=["auto"],
            speakers=["ghost"],
            non_streaming_mode=False,
        )
    with pytest.raises(NotImplementedError):
        gen_model.generate(
            input_ids=[
                torch.tensor([[1, 2, 3, 14, 15, 16, 17, 18, 19]], dtype=torch.long)
            ],
            languages=["klingon"],
            speakers=[None],
            non_streaming_mode=False,
        )
