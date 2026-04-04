from __future__ import annotations


from types import SimpleNamespace

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


def test_qwen_tts_modeling_tokenizer_v2_paths(monkeypatch) -> None:
    _ensure_transformers_patch(monkeypatch)

    import src.lib.audio.qwen_tts.tts_model.configuration_qwen3_tts_tokenizer_v2 as cfg_mod
    import src.lib.audio.qwen_tts.tts_model.modeling_qwen3_tts_tokenizer_v2 as tokv2_mod

    monkeypatch.setitem(
        tokv2_mod.ROPE_INIT_FUNCTIONS,
        "default",
        lambda config, device=None: (
            torch.ones(config.head_dim // 2, dtype=torch.float32),
            1.0,
        ),
    )

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(tokv2_mod.rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    q = torch.ones((1, 1, 2, 4))
    k = torch.ones((1, 1, 2, 4))
    cos = torch.ones((1, 2, 4))
    sin = torch.zeros((1, 2, 4))
    q2, k2 = tokv2_mod.apply_rotary_pos_emb(q, k, cos, sin)
    assert q2.shape == q.shape and k2.shape == k.shape

    kv = torch.ones((1, 1, 2, 4))
    assert tokv2_mod.repeat_kv(kv, 1).shape == kv.shape
    assert tokv2_mod.repeat_kv(kv, 2).shape == (1, 2, 2, 4)

    dummy_module = SimpleNamespace(num_key_value_groups=1, training=False)
    attn_out, attn_weights = tokv2_mod.eager_attention_forward(
        dummy_module,
        q,
        k,
        torch.ones_like(k),
        torch.zeros((1, 1, 2, 2)),
        scaling=0.5,
    )
    assert attn_out.shape == (1, 2, 1, 4)
    assert attn_weights.shape == (1, 1, 2, 2)

    conv = tokv2_mod.Qwen3TTSTokenizerV2CausalConvNet(1, 1, kernel_size=3, stride=2)
    hidden = torch.ones((1, 1, 5))
    assert conv._get_extra_padding_for_conv1d(hidden) >= 0
    assert conv(hidden).shape[0] == 1

    trans_conv = tokv2_mod.Qwen3TTSTokenizerV2CausalTransConvNet(
        1, 1, kernel_size=4, stride=2
    )
    assert trans_conv(torch.ones((1, 1, 4))).shape[0] == 1

    convnext = tokv2_mod.Qwen3TTSTokenizerV2ConvNeXtBlock(4)
    assert convnext(torch.ones((1, 4, 8))).shape == (1, 4, 8)

    cfg = cfg_mod.Qwen3TTSTokenizerV2DecoderConfig(
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
    cfg._attn_implementation = "eager"

    rotary = tokv2_mod.Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(cfg)
    pos_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    cos2, sin2 = rotary(torch.ones((1, 3, 4)), pos_ids)
    assert cos2.shape == (1, 3, 4)
    assert sin2.shape == (1, 3, 4)

    attention = tokv2_mod.Qwen3TTSTokenizerV2DecoderAttention(cfg, layer_idx=0)

    class _Cache:
        def update(self, key_states, value_states, layer_idx, cache_kwargs):
            return key_states, value_states

    hidden_states = torch.ones((1, 3, 4))
    mask = torch.zeros((1, 1, 3, 3))
    attn_output, attn_w = attention(
        hidden_states,
        (cos2, sin2),
        mask,
        past_key_values=_Cache(),
        cache_position=torch.tensor([0, 1, 2]),
    )
    assert attn_output.shape == (1, 3, 4)
    assert attn_w is not None

    cfg_alt = cfg_mod.Qwen3TTSTokenizerV2DecoderConfig(
        hidden_size=4,
        latent_dim=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        num_hidden_layers=1,
        num_quantizers=2,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
        decoder_dim=4,
        head_dim=4,
        codebook_dim=4,
        codebook_size=8,
    )
    cfg_alt._attn_implementation = "custom"
    monkeypatch.setitem(
        tokv2_mod.ALL_ATTENTION_FUNCTIONS,
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
    attention_alt = tokv2_mod.Qwen3TTSTokenizerV2DecoderAttention(cfg_alt, layer_idx=0)
    alt_out, _ = attention_alt(hidden_states, (cos2, sin2), mask)
    assert alt_out.shape == (1, 3, 4)

    mlp = tokv2_mod.Qwen3TTSTokenizerV2DecoderMlp(cfg)
    assert mlp(torch.ones((1, 3, 4))).shape == (1, 3, 4)

    rms = tokv2_mod.Qwen3TTSTokenizerV2DecoderRMSNorm(4)
    assert rms(torch.ones((1, 3, 4))).shape == (1, 3, 4)
    assert "eps=" in rms.extra_repr()

    layer_scale = tokv2_mod.Qwen3TTSTokenizerV2DecoderLayerScale(cfg)
    assert layer_scale(torch.ones((1, 3, 4))).shape == (1, 3, 4)

    transformer_layer = tokv2_mod.Qwen3TTSTokenizerV2DecoderTransformerLayer(
        cfg, layer_idx=0
    )
    with pytest.raises(TypeError):
        transformer_layer(
            hidden_states,
            attention_mask=mask,
            position_ids=pos_ids,
            cache_position=torch.tensor([0, 1, 2]),
        )
    monkeypatch.setattr(
        transformer_layer.self_attn,
        "forward",
        lambda **kwargs: (kwargs["hidden_states"], None),
    )
    layer_out = transformer_layer(
        hidden_states,
        attention_mask=mask,
        position_ids=pos_ids,
        cache_position=torch.tensor([0, 1, 2]),
    )
    assert layer_out.shape == (1, 3, 4)

    transformer_model = tokv2_mod.Qwen3TTSTokenizerV2DecoderTransformerModel(cfg)
    for layer in transformer_model.layers:
        monkeypatch.setattr(
            layer.self_attn,
            "forward",
            lambda **kwargs: (kwargs["hidden_states"], None),
        )
    model_out = transformer_model(inputs_embeds=torch.ones((1, 3, 4)))
    assert model_out.last_hidden_state.shape == (1, 3, 4)
    with pytest.raises(ValueError):
        transformer_model(
            input_ids=torch.ones((1, 3), dtype=torch.long),
            inputs_embeds=torch.ones((1, 3, 4)),
        )

    cached_out = transformer_model(
        inputs_embeds=torch.ones((1, 3, 4)),
        use_cache=True,
    )
    assert cached_out.past_key_values is not None

    snake = tokv2_mod.SnakeBeta(4)
    assert snake(torch.ones((1, 4, 3))).shape == (1, 4, 3)

    residual_unit = tokv2_mod.Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(
        dim=4, dilation=1
    )
    assert residual_unit(torch.ones((1, 4, 8))).shape == (1, 4, 8)

    decoder_block = tokv2_mod.Qwen3TTSTokenizerV2DecoderDecoderBlock(cfg, layer_idx=0)
    assert decoder_block(torch.ones((1, 4, 8))).shape[1] == 2

    codebook = tokv2_mod.EuclideanCodebook(dim=2, codebook_size=4)
    assert codebook.decode(torch.tensor([[0, 1]], dtype=torch.long)).shape == (1, 2, 2)

    vq = tokv2_mod.VectorQuantization(dim=2, codebook_size=4, codebook_dim=1)
    assert vq.decode(torch.tensor([[0, 1]], dtype=torch.long)).shape == (1, 2, 2)

    rvq = tokv2_mod.ResidualVectorQuantization(num_quantizers=2, dim=2, codebook_size=4)
    assert rvq.decode(torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.long)).shape == (
        1,
        2,
        2,
    )

    rvq_proj = tokv2_mod.ResidualVectorQuantizer(
        dimension=2,
        input_dimension=3,
        output_dimension=4,
        n_q=2,
        bins=4,
        force_projection=True,
    )
    assert rvq_proj.decode(
        torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    ).shape == (1, 4, 2)
    rvq_identity = tokv2_mod.ResidualVectorQuantizer(
        dimension=2,
        input_dimension=2,
        output_dimension=2,
        n_q=2,
        bins=4,
    )
    assert isinstance(rvq_identity.input_proj, torch.nn.Identity)
    assert isinstance(rvq_identity.output_proj, torch.nn.Identity)

    split = tokv2_mod.SplitResidualVectorQuantizer(
        n_q=3,
        n_q_semantic=1,
        dimension=2,
        input_dimension=2,
        output_dimension=2,
        bins=4,
    )
    assert split.decode(
        torch.tensor([[[0, 1], [1, 0], [0, 0]]], dtype=torch.long)
    ).shape == (1, 2, 2)

    decoder = tokv2_mod.Qwen3TTSTokenizerV2Decoder(cfg)
    for layer in decoder.pre_transformer.layers:
        monkeypatch.setattr(
            layer.self_attn,
            "forward",
            lambda **kwargs: (kwargs["hidden_states"], None),
        )
    codes = torch.randint(0, 8, (1, 2, 3), dtype=torch.long)
    wav = decoder(codes)
    assert wav.shape[0] == 1
    chunked = decoder.chunked_decode(codes, chunk_size=2, left_context_size=1)
    assert chunked.shape[0] == 1
    with pytest.raises(ValueError):
        decoder(torch.randint(0, 8, (1, 1, 3), dtype=torch.long))

    fake_encoder = SimpleNamespace(
        encode=lambda input_values=None, return_dict=True: SimpleNamespace(
            audio_codes=torch.tensor([[[1, 2, 3], [3, 2, 1]]], dtype=torch.long)
        )
    )
    fake_decoder = SimpleNamespace(
        chunked_decode=lambda codes: torch.ones(
            (codes.shape[0], 1, codes.shape[-1] * 2), dtype=torch.float32
        )
    )
    monkeypatch.setattr(
        tokv2_mod.Qwen3TTSTokenizerV2Encoder,
        "_from_config",
        classmethod(lambda cls, config: fake_encoder),
    )
    monkeypatch.setattr(
        tokv2_mod.Qwen3TTSTokenizerV2Decoder,
        "_from_config",
        classmethod(lambda cls, config: fake_decoder),
    )

    wrapper_cfg = cfg_mod.Qwen3TTSTokenizerV2Config(
        decoder_config={
            "num_hidden_layers": 1,
            "hidden_size": 4,
            "latent_dim": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_quantizers": 2,
            "codebook_size": 8,
            "upsample_rates": (2,),
            "upsampling_ratios": (2,),
            "decoder_dim": 4,
            "codebook_dim": 4,
            "head_dim": 4,
        },
        encode_downsample_rate=2,
        decode_upsample_rate=2,
    )
    wrapper = tokv2_mod.Qwen3TTSTokenizerV2Model(wrapper_cfg)
    assert wrapper.get_model_type() == wrapper_cfg.model_type
    assert wrapper.get_input_sample_rate() == wrapper_cfg.input_sample_rate
    assert wrapper.get_output_sample_rate() == wrapper_cfg.output_sample_rate
    assert wrapper.get_encode_downsample_rate() == 2
    assert wrapper.get_decode_upsample_rate() == 2

    input_values = torch.ones((1, 6), dtype=torch.float32)
    padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    encoded = wrapper.encode(input_values, padding_mask, return_dict=True)
    assert len(encoded.audio_codes) == 1
    encoded_tuple = wrapper.encode(input_values, padding_mask, return_dict=False)
    assert len(encoded_tuple[0]) == 1

    decoded = wrapper.decode(
        torch.tensor([[[1, 2], [3, 4], [1, 0]]], dtype=torch.long), return_dict=True
    )
    assert len(decoded.audio_values) == 1
    decoded_tuple = wrapper.decode(
        torch.tensor([[[1, 2], [3, 4], [1, 0]]], dtype=torch.long), return_dict=False
    )
    assert len(decoded_tuple[0]) == 1

    monkeypatch.setattr(tokv2_mod.MimiModel, "__init__", lambda self, config: None)
    monkeypatch.setattr(
        tokv2_mod.Qwen3TTSTokenizerV2Encoder,
        "post_init",
        lambda self: setattr(self, "post_inited", True),
    )
    encoder = tokv2_mod.Qwen3TTSTokenizerV2Encoder(
        cfg_mod.MimiConfig(hidden_size=4, num_hidden_layers=1)
    )
    assert encoder.post_inited is True


def test_qwen_tts_modeling_tokenizer_v2_remaining_paths(monkeypatch) -> None:
    _ensure_transformers_patch(monkeypatch)

    import src.lib.audio.qwen_tts.tts_model.configuration_qwen3_tts_tokenizer_v2 as cfg_mod
    import src.lib.audio.qwen_tts.tts_model.modeling_qwen3_tts_tokenizer_v2 as tokv2_mod

    monkeypatch.setitem(
        tokv2_mod.ROPE_INIT_FUNCTIONS,
        "default",
        lambda config, device=None: (
            torch.ones(config.head_dim // 2, dtype=torch.float32),
            1.0,
        ),
    )

    cfg = cfg_mod.Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=8,
        hidden_size=4,
        latent_dim=4,
        max_position_embeddings=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        num_hidden_layers=1,
        num_quantizers=2,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
        decoder_dim=4,
        head_dim=4,
        codebook_dim=4,
        rope_theta=10000,
    )
    cfg._attn_implementation = "eager"
    cfg.rope_scaling = {"type": "default"}

    rotary = tokv2_mod.Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(cfg)
    assert rotary.rope_type == "default"

    transformer = tokv2_mod.Qwen3TTSTokenizerV2DecoderTransformerModel(cfg)
    with pytest.raises(ValueError):
        transformer()


def test_qwen_tts_modeling_tokenizer_v2_check_model_inputs_decorator(
    monkeypatch,
) -> None:
    import src.lib.audio.qwen_tts.tts_model.modeling_qwen3_tts_tokenizer_v2 as tokv2_mod

    def _sample_func():
        return "ok"

    decorator = tokv2_mod.check_model_inputs()
    wrapped = decorator(_sample_func)
    assert callable(wrapped)
    assert wrapped() == "ok"

    monkeypatch.setattr(tokv2_mod, "check_model_inputs", lambda: lambda func: func)
    assert tokv2_mod.check_model_inputs()(_sample_func) is _sample_func
