from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from src.lib.motion.dart_control.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    _extract_into_tensor,
    betas_for_alpha_bar,
    get_named_beta_schedule,
)


class _Model(nn.Module):
    def __init__(self, out_channels: int = 1, num_classes: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.calls: list[tuple[torch.Tensor, dict[str, object]]] = []

    def forward(self, x, ts, **kwargs):
        self.calls.append((ts.clone(), dict(kwargs)))
        shape = (x.shape[0], self.out_channels, *x.shape[2:])
        return torch.ones(shape, device=x.device, dtype=x.dtype)


def _make_diffusion(
    *,
    model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
    model_var_type: ModelVarType = ModelVarType.FIXED_SMALL,
    rescale_timesteps: bool = False,
) -> GaussianDiffusion:
    return GaussianDiffusion(
        betas=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=LossType.MSE,
        rescale_timesteps=rescale_timesteps,
    )


def test_gaussian_diffusion_schedule_and_enum_helpers() -> None:
    linear = get_named_beta_schedule("linear", 4)
    cosine = get_named_beta_schedule("cosine", 4)
    custom = betas_for_alpha_bar(4, lambda t: 1.0 - (0.25 * t), max_beta=0.5)

    assert linear.shape == (4,)
    assert cosine.shape == (4,)
    assert custom.shape == (4,)
    assert LossType.KL.is_vb() is True
    assert LossType.RESCALED_KL.is_vb() is True
    assert LossType.MSE.is_vb() is False

    with pytest.raises(NotImplementedError):
        get_named_beta_schedule("bad", 4)


def test_extract_and_basic_q_functions() -> None:
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    timesteps = torch.tensor([0, 2], dtype=torch.long)
    extracted = _extract_into_tensor(arr, timesteps, (2, 1, 1))
    assert extracted.shape == (2, 1, 1)

    diffusion = _make_diffusion(rescale_timesteps=True)
    x_start = torch.ones((2, 1, 1, 1), dtype=torch.float32)
    t = torch.tensor([0, 1], dtype=torch.long)
    mean, variance, log_variance = diffusion.q_mean_variance(x_start, t)
    assert mean.shape == variance.shape == log_variance.shape == x_start.shape

    sample = diffusion.q_sample(x_start, t, noise=torch.zeros_like(x_start))
    assert sample.shape == x_start.shape
    sampled_default = diffusion.q_sample(x_start, t)
    assert sampled_default.shape == x_start.shape

    with pytest.raises(AssertionError):
        diffusion.q_sample(x_start, t, noise=torch.zeros((1, 1, 1, 1)))

    posterior = diffusion.q_posterior_mean_variance(x_start, sample, t)
    assert all(part.shape == x_start.shape for part in posterior)
    assert torch.allclose(
        diffusion._scale_timesteps(torch.tensor([1])),
        torch.tensor([333.3333], dtype=torch.float32),
        atol=1e-3,
    )


def test_p_mean_variance_and_predictor_branches() -> None:
    x = torch.ones((2, 1, 1, 1), dtype=torch.float32)
    t = torch.tensor([1, 2], dtype=torch.long)

    diffusion_eps = _make_diffusion(model_mean_type=ModelMeanType.EPSILON)
    model = _Model(out_channels=1)
    out_eps = diffusion_eps.p_mean_variance(
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=lambda value: value + 2,
        model_kwargs={"flag": True},
    )
    assert out_eps["mean"].shape == x.shape

    diffusion_start = _make_diffusion(model_mean_type=ModelMeanType.START_X)
    out_start = diffusion_start.p_mean_variance(model, x, t)
    assert out_start["pred_xstart"].shape == x.shape

    diffusion_prev = _make_diffusion(model_mean_type=ModelMeanType.PREVIOUS_X)
    out_prev = diffusion_prev.p_mean_variance(model, x, t)
    assert out_prev["pred_xstart"].shape == x.shape

    learned = _make_diffusion(model_var_type=ModelVarType.LEARNED)
    learned_model = _Model(out_channels=2)
    out_learned = learned.p_mean_variance(learned_model, x, t)
    assert out_learned["variance"].shape == x.shape

    learned_range = _make_diffusion(model_var_type=ModelVarType.LEARNED_RANGE)
    out_range = learned_range.p_mean_variance(learned_model, x, t)
    assert out_range["log_variance"].shape == x.shape

    fixed_large = _make_diffusion(model_var_type=ModelVarType.FIXED_LARGE)
    out_large = fixed_large.p_mean_variance(model, x, t)
    assert out_large["variance"].shape == x.shape

    bad = _make_diffusion()
    bad.model_mean_type = "bad"  # type: ignore[assignment]
    with pytest.raises(NotImplementedError):
        bad.p_mean_variance(model, x, t)

    pred_xstart = diffusion_eps._predict_xstart_from_eps(x, t, torch.zeros_like(x))
    pred_xprev = diffusion_eps._predict_xstart_from_xprev(x, t, torch.zeros_like(x))
    pred_eps = diffusion_eps._predict_eps_from_xstart(x, t, torch.zeros_like(x))
    assert pred_xstart.shape == pred_xprev.shape == pred_eps.shape == x.shape


def test_sampling_loops_and_progress_paths(monkeypatch) -> None:
    diffusion = _make_diffusion()
    model = _Model(out_channels=1)
    x = torch.ones((2, 1, 1, 1), dtype=torch.float32)
    t = torch.tensor([1, 0], dtype=torch.long)

    out = diffusion.p_sample(model, x, t, const_noise=True)
    assert out["sample"].shape == x.shape

    monkeypatch.setattr(
        diffusion,
        "p_sample",
        lambda *args, **kwargs: {
            "sample": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            "pred_xstart": torch.ones((2, 1, 1, 1), dtype=torch.float32),
        },
    )

    fake_tqdm = types.SimpleNamespace(tqdm=lambda seq: list(seq))
    monkeypatch.setitem(sys.modules, "tqdm.auto", fake_tqdm)

    progressive = list(
        diffusion.p_sample_loop_progressive(
            model,
            shape=(2, 1, 1, 1),
            noise=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            progress=True,
            skip_timesteps=1,
            init_image=torch.ones((2, 1, 1, 1), dtype=torch.float32),
            randomize_class=True,
            model_kwargs={"y": torch.zeros((2,), dtype=torch.long)},
            const_noise=True,
        )
    )
    assert progressive

    final = diffusion.p_sample_loop(
        model,
        shape=(2, 1, 1, 1),
        noise=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
    )
    dumped = diffusion.p_sample_loop(
        model,
        shape=(2, 1, 1, 1),
        noise=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        dump_steps=[0],
    )
    seeded = list(
        diffusion.p_sample_loop_progressive(
            model,
            shape=(2, 1, 1, 1),
            noise=None,
            skip_timesteps=1,
            init_image=None,
        )
    )
    assert final.shape == (2, 1, 1, 1)
    assert len(dumped) == 1
    assert seeded


def test_ddim_sampling_paths(monkeypatch) -> None:
    diffusion = _make_diffusion()
    model = _Model(out_channels=1)
    x = torch.ones((2, 1, 1, 1), dtype=torch.float32)
    t = torch.tensor([1, 0], dtype=torch.long)

    out = diffusion.ddim_sample(model, x, t, eta=0.5)
    assert out["sample"].shape == x.shape

    monkeypatch.setattr(
        diffusion,
        "ddim_sample",
        lambda *args, **kwargs: {
            "sample": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            "pred_xstart": torch.ones((2, 1, 1, 1), dtype=torch.float32),
        },
    )

    fake_tqdm = types.SimpleNamespace(tqdm=lambda seq: list(seq))
    monkeypatch.setitem(sys.modules, "tqdm.auto", fake_tqdm)

    progressive = list(
        diffusion.ddim_sample_loop_progressive(
            model,
            shape=(2, 1, 1, 1),
            noise=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
            progress=True,
            skip_timesteps=1,
            init_image=torch.ones((2, 1, 1, 1), dtype=torch.float32),
            randomize_class=True,
            model_kwargs={"y": torch.zeros((2,), dtype=torch.long)},
            eta=0.2,
        )
    )
    assert progressive

    final = diffusion.ddim_sample_loop(
        model,
        shape=(2, 1, 1, 1),
        noise=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        eta=0.1,
    )
    seeded = list(
        diffusion.ddim_sample_loop_progressive(
            model,
            shape=(2, 1, 1, 1),
            noise=None,
            skip_timesteps=1,
            init_image=None,
        )
    )
    assert final.shape == (2, 1, 1, 1)
    assert seeded
