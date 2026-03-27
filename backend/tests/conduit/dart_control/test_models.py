from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from torch import nn


def test_dart_control_operators_paths() -> None:
    import src.core.conduit.dart_control.models.operators as ops_mod

    linear = nn.Linear(2, 2)
    cloned = ops_mod._get_clone(linear)
    clones = ops_mod._get_clones(linear, 3)
    assert cloned is not linear
    assert len(clones) == 3

    assert ops_mod._get_activation_fn("relu") is torch.nn.functional.relu
    assert ops_mod._get_activation_fn("gelu") is torch.nn.functional.gelu
    assert ops_mod._get_activation_fn("glu") is torch.nn.functional.glu
    with pytest.raises(RuntimeError):
        ops_mod._get_activation_fn("bad")

    pos_sine = ops_mod.PositionEmbeddingSine1D(4, max_len=8, batch_first=False)
    x = torch.zeros((3, 2, 4))
    assert pos_sine(x).shape == (3, 1, 4)
    with pytest.raises(UnboundLocalError):
        ops_mod.PositionEmbeddingSine1D(4, max_len=8, batch_first=True)(torch.zeros((2, 3, 4)))

    pos_learned = ops_mod.PositionEmbeddingLearned1D(4, max_len=8, batch_first=False)
    out = pos_learned(x.clone())
    assert out.shape == x.shape
    out_bf = ops_mod.PositionEmbeddingLearned1D(4, max_len=8, batch_first=True)(torch.zeros((2, 3, 4)))
    assert out_bf.shape == (2, 3, 4)

    assert isinstance(ops_mod.build_position_encoding(4, "sine"), ops_mod.PositionEmbeddingSine1D)
    assert isinstance(ops_mod.build_position_encoding(4, "learned"), ops_mod.PositionEmbeddingLearned1D)
    with pytest.raises(ValueError):
        ops_mod.build_position_encoding(4, "bad")
    with pytest.raises(ValueError):
        ops_mod.build_position_encoding(4, "sine", embedding_dim="2D")

    pe = ops_mod.PositionalEncoding(4, dropout=0.0, max_len=8)
    assert pe(torch.zeros((3, 2, 4))).shape == (3, 2, 4)

    src = torch.randn(3, 2, 4)
    pos = torch.randn(3, 2, 4)
    enc_post = ops_mod.TransformerEncoderLayer(4, 1, dim_feedforward=8, dropout=0.0, normalize_before=False)
    enc_pre = ops_mod.TransformerEncoderLayer(4, 1, dim_feedforward=8, dropout=0.0, normalize_before=True)
    assert enc_post.with_pos_embed(src, None).shape == src.shape
    assert enc_post.forward_post(src, pos=pos).shape == src.shape
    assert enc_pre.forward_pre(src, pos=pos).shape == src.shape
    assert enc_post(src, pos=pos).shape == src.shape
    assert enc_pre(src, pos=pos).shape == src.shape

    tgt = torch.randn(3, 2, 4)
    memory = torch.randn(3, 2, 4)
    dec_post = ops_mod.TransformerDecoderLayer(4, 1, dim_feedforward=8, dropout=0.0, normalize_before=False)
    dec_pre = ops_mod.TransformerDecoderLayer(4, 1, dim_feedforward=8, dropout=0.0, normalize_before=True)
    assert dec_post.forward_post(tgt, memory, pos=pos, query_pos=pos).shape == tgt.shape
    assert dec_pre.forward_pre(tgt, memory, pos=pos, query_pos=pos).shape == tgt.shape
    assert dec_post(tgt, memory, pos=pos, query_pos=pos).shape == tgt.shape
    assert dec_pre(tgt, memory, pos=pos, query_pos=pos).shape == tgt.shape

    with pytest.raises(AssertionError):
        ops_mod.SkipTransformerEncoder(enc_post, 2)
    skip_enc = ops_mod.SkipTransformerEncoder(enc_post, 3, norm=nn.LayerNorm(4))
    assert skip_enc(src, pos=pos).shape == src.shape

    with pytest.raises(AssertionError):
        ops_mod.SkipTransformerDecoder(dec_post, 2)
    skip_dec = ops_mod.SkipTransformerDecoder(dec_post, 3, norm=nn.LayerNorm(4))
    assert skip_dec(tgt, memory, pos=pos, query_pos=pos).shape == tgt.shape

    tsteps = torch.tensor([1, 2], dtype=torch.long)
    emb = ops_mod.get_timestep_embedding(tsteps, 5, flip_sin_to_cos=True)
    assert emb.shape == (2, 5)
    with pytest.raises(AssertionError):
        ops_mod.get_timestep_embedding(torch.ones((2, 2)), 4)

    te = ops_mod.TimestepEmbedding(4, 6, act_fn="silu")
    assert te(torch.ones((2, 4))).shape == (2, 6)
    te_noact = ops_mod.TimestepEmbedding(4, 6, act_fn="none")
    assert te_noact(torch.ones((2, 4))).shape == (2, 6)
    assert ops_mod.Timesteps(6, flip_sin_to_cos=True)(tsteps).shape == (2, 6)


def test_dart_control_denoiser_paths() -> None:
    import src.core.conduit.dart_control.models.denoiser as den_mod

    seq_pos = den_mod.PositionalEncoding(4, dropout=0.0, max_len=8)
    t_embed = den_mod.TimestepEmbedder(4, seq_pos)
    assert t_embed(torch.tensor([1, 2], dtype=torch.long)).shape == (1, 2, 4)

    for act in ["tanh", "relu", "sigmoid", "gelu", "lrelu", "bad"]:
        mlp = den_mod.MLP(4, h_dims=(4, 4), activation=act)
        assert mlp(torch.ones((2, 4))).shape == (2, 4)

    assert den_mod.MLPBlock(4, 2, 2, residual=True)(torch.ones((2, 4))).shape == (2, 2)
    assert den_mod.MLPBlock(4, 2, 2, residual=False)(torch.ones((2, 4))).shape == (2, 2)

    y = {
        "text_embedding": torch.ones((2, 4)),
        "history_motion_normalized": torch.ones((2, 2, 3)),
        "scale": torch.full((2, 1, 3), 2.0),
    }
    x_t = torch.ones((2, 1, 3))
    timesteps = torch.tensor([1, 2], dtype=torch.long)

    den_mlp = den_mod.DenoiserMLP(
        h_dim=4,
        n_blocks=2,
        dropout=0.0,
        activation="relu",
        clip_dim=4,
        history_shape=(2, 3),
        noise_shape=(1, 3),
        cond_mask_prob=0.5,
    )
    assert den_mlp.parameters_wo_clip()
    assert torch.count_nonzero(den_mlp.mask_cond(torch.ones((2, 4)), force_mask=True)) == 0
    den_mlp.train()
    masked = den_mlp.mask_cond(torch.ones((2, 4)))
    assert masked.shape == (2, 4)
    den_mlp.eval()
    assert den_mlp.mask_cond(torch.ones((2, 4))).shape == (2, 4)
    assert den_mlp(x_t, timesteps, y).shape == (2, 1, 3)

    den_tf = den_mod.DenoiserTransformer(
        h_dim=4,
        ff_size=8,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
        activation="relu",
        clip_dim=4,
        history_shape=(2, 3),
        noise_shape=(1, 3),
        cond_mask_prob=0.5,
    )
    assert den_tf.parameters_wo_clip()
    assert torch.count_nonzero(den_tf.mask_cond(torch.ones((2, 4)), force_mask=True)) == 0
    den_tf.train()
    den_tf.mask_cond(torch.ones((2, 4)))
    den_tf.eval()
    assert den_tf(x_t, timesteps, y).shape == (2, 1, 3)

    with pytest.raises(AssertionError):
        den_mod.ClassifierFreeWrapper(
            SimpleNamespace(cond_mask_prob=0, __call__=lambda *args, **kwargs: torch.zeros((2, 1, 3)))
        )

    wrapper = den_mod.ClassifierFreeWrapper(den_tf)
    guided = wrapper(x_t, timesteps, dict(y))
    assert guided.shape == (2, 1, 3)


def test_dart_control_vae_paths() -> None:
    import src.core.conduit.dart_control.models.vae as vae_mod

    with pytest.raises(ValueError):
        vae_mod.AutoMldVae(nfeats=3, latent_dim=(1, 4), h_dim=4, ff_size=8, num_layers=3, num_heads=1, dropout=0.0, arch="bad")

    history = torch.ones((2, 2, 3), dtype=torch.float32)
    future = torch.ones((2, 3, 3), dtype=torch.float32)

    vae_all = vae_mod.AutoMldVae(
        nfeats=3,
        latent_dim=(1, 4),
        h_dim=4,
        ff_size=8,
        num_layers=3,
        num_heads=1,
        dropout=0.0,
        arch="all_encoder",
        position_embedding="learned",
    )
    latent, dist = vae_all.encode(future, history, scale_latent=True)
    assert latent.shape == (1, 2, 4)
    assert hasattr(dist, "rsample")
    decoded = vae_all.decode(latent, history, nfuture=3, scale_latent=True)
    assert decoded.shape == (2, 3, 3)

    vae_ed = vae_mod.AutoMldVae(
        nfeats=3,
        latent_dim=(1, 4),
        h_dim=4,
        ff_size=8,
        num_layers=3,
        num_heads=1,
        dropout=0.0,
        arch="encoder_decoder",
        position_embedding="sine",
    )
    latent2, _ = vae_ed.encode(future, history, scale_latent=False)
    decoded2 = vae_ed.decode(latent2, history, nfuture=2, scale_latent=False)
    assert decoded2.shape[-2:] == (2, 3)
