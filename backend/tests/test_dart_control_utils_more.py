from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


def test_diffusion_losses_nn_and_respace(monkeypatch) -> None:
    import src.core.conduit.dart_control.diffusion.losses as losses_mod
    import src.core.conduit.dart_control.diffusion.nn as nn_mod
    import src.core.conduit.dart_control.diffusion.respace as respace_mod
    from src.core.conduit.dart_control.diffusion.gaussian_diffusion import (
        LossType,
        ModelMeanType,
        ModelVarType,
    )

    mean1 = torch.tensor([0.0, 1.0], dtype=torch.float32)
    kl = losses_mod.normal_kl(mean1, 0.0, torch.tensor([1.0, 0.0]), 0.0)
    assert kl.shape == (2,)
    with pytest.raises(AssertionError):
        losses_mod.normal_kl(0.0, 0.0, 0.0, 0.0)

    cdf = losses_mod.approx_standard_normal_cdf(torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.all((cdf > 0) & (cdf < 1))

    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    ll = losses_mod.discretized_gaussian_log_likelihood(
        x,
        means=torch.zeros_like(x),
        log_scales=torch.zeros_like(x),
    )
    assert ll.shape == x.shape

    silu = nn_mod.SiLU()
    assert torch.allclose(silu(torch.tensor([0.0, 1.0])), torch.tensor([0.0, torch.sigmoid(torch.tensor(1.0)).item()]))

    gn = nn_mod.GroupNorm32(1, 1)
    x32 = gn(torch.ones((1, 1, 2), dtype=torch.float16))
    assert x32.dtype == torch.float16

    assert isinstance(nn_mod.conv_nd(1, 1, 1, 1), nn.Conv1d)
    assert isinstance(nn_mod.conv_nd(2, 1, 1, 1), nn.Conv2d)
    assert isinstance(nn_mod.conv_nd(3, 1, 1, 1), nn.Conv3d)
    with pytest.raises(ValueError):
        nn_mod.conv_nd(4, 1, 1, 1)

    assert isinstance(nn_mod.linear(2, 3), nn.Linear)
    assert isinstance(nn_mod.avg_pool_nd(1, 1), nn.AvgPool1d)
    assert isinstance(nn_mod.avg_pool_nd(2, 1), nn.AvgPool2d)
    assert isinstance(nn_mod.avg_pool_nd(3, 1), nn.AvgPool3d)
    with pytest.raises(ValueError):
        nn_mod.avg_pool_nd(4, 1)

    targ = [torch.nn.Parameter(torch.tensor([1.0]))]
    src = [torch.nn.Parameter(torch.tensor([3.0]))]
    nn_mod.update_ema(targ, src, rate=0.5)
    assert torch.allclose(targ[0], torch.tensor([2.0]))

    linear = nn.Linear(2, 2)
    nn.init.ones_(linear.weight)
    nn_mod.scale_module(linear, 0.5)
    assert torch.allclose(linear.weight, torch.full_like(linear.weight, 0.5))
    nn_mod.zero_module(linear)
    assert torch.count_nonzero(linear.weight) == 0

    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.allclose(nn_mod.mean_flat(tensor), tensor.mean(dim=(1, 2)))
    assert torch.allclose(nn_mod.sum_flat(tensor), tensor.sum(dim=(1, 2)))
    assert isinstance(nn_mod.normalization(32), nn_mod.GroupNorm32)

    emb_even = nn_mod.timestep_embedding(torch.tensor([0, 1]), dim=4)
    emb_odd = nn_mod.timestep_embedding(torch.tensor([0, 1]), dim=5)
    assert emb_even.shape == (2, 4)
    assert emb_odd.shape == (2, 5)

    x_in = torch.tensor([2.0], requires_grad=True)
    assert torch.allclose(nn_mod.checkpoint(lambda t: t + 1, [x_in], [], False), torch.tensor([3.0]))
    out = nn_mod.checkpoint(lambda t: t * t, [x_in], [], True)
    out.sum().backward()
    assert torch.allclose(x_in.grad, torch.tensor([4.0]))

    assert respace_mod.space_timesteps(10, [1, 2]) == {0, 5, 9}
    assert len(respace_mod.space_timesteps(10, "ddim5")) == 5
    assert len(respace_mod.space_timesteps(9, "1,1,1")) == 3
    with pytest.raises(ValueError):
        respace_mod.space_timesteps(4, [5])
    with pytest.raises(ValueError):
        respace_mod.space_timesteps(4, "ddim7")

    monkeypatch.setattr(
        respace_mod.GaussianDiffusion,
        "p_mean_variance",
        lambda self, model, *args, **kwargs: ("wrapped", isinstance(model, respace_mod._WrappedModel)),
    )
    spaced = respace_mod.SpacedDiffusion(
        use_timesteps={0, 2},
        betas=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=True,
    )
    class _WrappedBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 1))

        def forward(self, x, ts, **kwargs):
            return x + ts.unsqueeze(-1).float()

    base_model = _WrappedBase()
    wrapped = spaced._wrap_model(base_model)
    assert spaced._wrap_model(wrapped) is wrapped
    wrapped_out = wrapped(torch.ones((2, 1)), torch.tensor([0, 1], dtype=torch.long))
    assert wrapped_out.shape == (2, 1)
    assert list(wrapped.parameters()) == list(base_model.parameters())
    assert torch.equal(spaced._scale_timesteps(torch.tensor([1])), torch.tensor([1]))
    assert spaced.p_mean_variance(base_model)[0] == "wrapped"


def test_rotation_conversions() -> None:
    import src.core.conduit.dart_control.rotation_conversions as rot_mod

    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    matrix = rot_mod.quaternion_to_matrix(quat)
    assert matrix.shape == (1, 3, 3)
    assert torch.allclose(rot_mod.matrix_to_quaternion(matrix), quat)

    assert torch.equal(
        rot_mod._copysign(torch.tensor([1.0, -2.0]), torch.tensor([-1.0, -3.0])),
        torch.tensor([-1.0, -2.0]),
    )
    assert torch.equal(
        rot_mod._sqrt_positive_part(torch.tensor([-1.0, 0.0, 4.0])),
        torch.tensor([0.0, 0.0, 2.0]),
    )

    with pytest.raises(ValueError):
        rot_mod.matrix_to_quaternion(torch.ones(2, 2))

    angle = torch.tensor([0.0, np.pi / 2], dtype=torch.float32)
    assert rot_mod._axis_angle_rotation("X", angle).shape == (2, 3, 3)
    assert rot_mod._axis_angle_rotation("Y", angle).shape == (2, 3, 3)
    assert rot_mod._axis_angle_rotation("Z", angle).shape == (2, 3, 3)

    with pytest.raises(ValueError):
        rot_mod.euler_angles_to_matrix(torch.tensor(1.0), "XYZ")
    with pytest.raises(ValueError):
        rot_mod.euler_angles_to_matrix(torch.zeros(1, 3), "XXZ")
    with pytest.raises(ValueError):
        rot_mod.euler_angles_to_matrix(torch.zeros(1, 3), "XQZ")

    euler = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    matrix_xyz = rot_mod.euler_angles_to_matrix(euler, "XYZ")
    assert matrix_xyz.shape == (1, 3, 3)
    assert rot_mod.matrix_to_euler_angles(matrix_xyz, "XYZ").shape == (1, 3)
    assert rot_mod.matrix_to_euler_angles(torch.eye(3).unsqueeze(0), "ZXZ").shape == (1, 3)

    with pytest.raises(ValueError):
        rot_mod.matrix_to_euler_angles(torch.eye(3), "XXZ")
    with pytest.raises(ValueError):
        rot_mod.matrix_to_euler_angles(torch.eye(2), "XYZ")

    assert rot_mod._index_from_letter("X") == 0
    assert rot_mod._index_from_letter("Y") == 1
    assert rot_mod._index_from_letter("Z") == 2

    rand_q = rot_mod.random_quaternions(2)
    rand_r = rot_mod.random_rotations(2)
    one_r = rot_mod.random_rotation()
    assert rand_q.shape == (2, 4)
    assert rand_r.shape == (2, 3, 3)
    assert one_r.shape == (3, 3)

    assert torch.equal(
        rot_mod.standardize_quaternion(torch.tensor([[-1.0, 1.0, 0.0, 0.0]])),
        torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
    )

    qx = rot_mod.axis_angle_to_quaternion(torch.tensor([[np.pi / 2, 0.0, 0.0]], dtype=torch.float32))
    qy = rot_mod.axis_angle_to_quaternion(torch.tensor([[0.0, np.pi / 2, 0.0]], dtype=torch.float32))
    assert rot_mod.quaternion_raw_multiply(qx, qy).shape == (1, 4)
    assert rot_mod.quaternion_multiply(qx, qy).shape == (1, 4)
    assert rot_mod.quaternion_invert(qx).shape == (1, 4)

    point = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    rotated = rot_mod.quaternion_apply(qx, point)
    assert rotated.shape == (1, 3)
    with pytest.raises(ValueError):
        rot_mod.quaternion_apply(qx, torch.ones(1, 2))

    axis_angle = torch.tensor([[0.0, 0.0, np.pi / 2]], dtype=torch.float32)
    aa_matrix = rot_mod.axis_angle_to_matrix(axis_angle)
    assert aa_matrix.shape == (1, 3, 3)
    assert rot_mod.matrix_to_axis_angle(aa_matrix).shape == (1, 3)
    assert rot_mod.quaternion_to_axis_angle(rot_mod.axis_angle_to_quaternion(axis_angle)).shape == (1, 3)
    small = torch.tensor([[1e-8, 0.0, 0.0]], dtype=torch.float32)
    assert rot_mod.axis_angle_to_quaternion(small).shape == (1, 4)
    assert rot_mod.quaternion_to_axis_angle(torch.tensor([[1.0, 1e-8, 0.0, 0.0]], dtype=torch.float32)).shape == (1, 3)

    rot6d = rot_mod.matrix_to_rotation_6d(torch.eye(3).unsqueeze(0))
    assert rot6d.shape == (1, 6)
    assert rot_mod.rotation_6d_to_matrix(rot6d).shape == (1, 3, 3)


def test_policy_and_inference_more(monkeypatch, tmp_path: Path) -> None:
    import src.core.conduit.dart_control.inference as inf_mod
    import src.core.conduit.dart_control.policy as policy_mod

    for act in ["tanh", "relu", "sigmoid", "gelu", "lrelu"]:
        mlp = policy_mod.MLP(4, h_dims=[4], activation=act)
        assert mlp(torch.ones((1, 4))).shape == (1, 4)
    with pytest.raises(ValueError):
        policy_mod.MLP(4, activation="bad")

    block_res = policy_mod.MLPBlock(4, 2, 2, residual=True)
    block_nores = policy_mod.MLPBlock(4, 2, 2, residual=False)
    assert block_res(torch.ones((1, 4))).shape == (1, 2)
    assert block_nores(torch.ones((1, 4))).shape == (1, 2)

    cfg = policy_mod.PolicyConfig(
        latent_dim=8,
        motion_dim=6,
        text_dim=4,
        goal_dim=4,
        scene_dim=1,
        action_dim=3,
        pred_std=False,
        use_tanh_scale=True,
    )
    policy = policy_mod.PolicyReachLocationMLP(cfg)
    obs = torch.ones((2, 3 + 1 + 4 + 6 + 1), dtype=torch.float32)
    emb = policy.get_embedding(obs)
    action = policy.get_action_mean(obs)
    assert emb.shape == (2, 8)
    assert action.shape == (2, 3)

    cfg_std = policy_mod.PolicyConfig(
        latent_dim=8,
        motion_dim=6,
        text_dim=4,
        goal_dim=4,
        scene_dim=1,
        action_dim=3,
        pred_std=True,
        use_tanh_scale=False,
    )
    policy_std = policy_mod.PolicyReachLocationMLP(cfg_std)
    assert policy_std.get_action_mean(obs).shape == (2, 3)

    raw_path = tmp_path / "raw.yaml"
    raw_path.write_text("- 1\n- 2\n", encoding="utf-8")
    assert isinstance(inf_mod._load_tyro_yaml(raw_path, inf_mod.DataArgs), inf_mod.DataArgs)

    mlp_yaml = tmp_path / "mlp.yaml"
    mlp_yaml.write_text(
        """
denoiser_args:
  model_type: mlp
  model_args:
    h_dim: 16
    n_blocks: 3
    history_shape: [2, 10]
    noise_shape: [1, 5]
  diffusion_args:
    diffusion_steps: 4
    noise_schedule: linear
""",
        encoding="utf-8",
    )
    mld = inf_mod._load_tyro_yaml(mlp_yaml, inf_mod.MLDArgs)
    assert isinstance(mld.denoiser_args.model_args, inf_mod.DenoiserMLPArgs)

    mvae_yaml = tmp_path / "mvae.yaml"
    mvae_yaml.write_text(
        """
model_args:
  latent_dim: [1, 16]
  h_dim: 32
""",
        encoding="utf-8",
    )
    mvae = inf_mod._load_tyro_yaml(mvae_yaml, inf_mod.MVAEArgs)
    assert mvae.model_args.latent_dim == (1, 16)

    diff = inf_mod._create_diffusion(inf_mod.DiffusionArgs(diffusion_steps=4, noise_schedule="cosine"))
    diff_spaced = inf_mod._create_diffusion(
        inf_mod.DiffusionArgs(diffusion_steps=4, noise_schedule="cosine", respacing="ddim2")
    )
    assert diff.num_timesteps == 4
    assert diff_spaced.num_timesteps == 2

    class _FakeDenoiser:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, device):
            self.device = device
            return self

        def load_state_dict(self, state):
            self.state = state

        def eval(self):
            self.eval_called = True
            return self

        def parameters(self):
            return [torch.nn.Parameter(torch.zeros(1))]

    class _FakeVAE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, device):
            self.device = device
            return self

        def load_state_dict(self, state):
            self.state = state

        def eval(self):
            self.eval_called = True
            return self

        def parameters(self):
            return [torch.nn.Parameter(torch.zeros(1))]

        def decode(self, latent, history_motion, nfuture, scale_latent):
            return torch.ones((latent.shape[1], nfuture, 5), dtype=torch.float32)

    monkeypatch.setattr(
        inf_mod,
        "_load_tyro_yaml",
        lambda path, cls: inf_mod.MLDArgs(
            denoiser_args=inf_mod.DenoiserArgs(
                model_type="mlp",
                model_args=inf_mod.DenoiserMLPArgs(
                    h_dim=8,
                    n_blocks=2,
                    history_shape=(2, 5),
                    noise_shape=(1, 3),
                ),
            )
        )
        if cls is inf_mod.MLDArgs
        else inf_mod.MVAEArgs(model_args=inf_mod.VAEArgs(latent_dim=(1, 3), nfeats=5)),
    )
    monkeypatch.setattr(inf_mod, "DenoiserMLP", _FakeDenoiser)
    monkeypatch.setattr(inf_mod, "AutoMldVae", _FakeVAE)
    monkeypatch.setattr(inf_mod, "ClassifierFreeWrapper", lambda model: SimpleNamespace(model=model))
    monkeypatch.setattr(
        inf_mod.torch,
        "load",
        lambda path, map_location=None, weights_only=None: {"model_state_dict": {"w": torch.tensor([1.0])}},
    )
    dummy = object.__new__(inf_mod.DartControlInference)
    dummy.device = "cpu"
    mld_args, denoiser_model, mvae_args, vae_model = dummy._load_models(
        str(tmp_path / "denoiser.pt"),
        str(tmp_path / "vae.pt"),
    )
    assert isinstance(mld_args, inf_mod.MLDArgs)
    assert hasattr(denoiser_model, "model")
    assert isinstance(mvae_args, inf_mod.MVAEArgs)
    assert torch.equal(vae_model.latent_mean, torch.tensor(0))
    assert torch.equal(vae_model.latent_std, torch.tensor(1))

    infer = object.__new__(inf_mod.DartControlInference)
    infer.device = "cpu"
    infer._mean = torch.ones((1, 1, 5))
    infer._std = torch.full((1, 1, 5), 2.0)
    infer.noise_shape = (1, 3)
    infer.rescale_latent = True
    infer._clip_model = SimpleNamespace(
        encode_text=lambda tokens: torch.arange(tokens.shape[0] * 4, dtype=torch.float32).reshape(tokens.shape[0], 4)
    )
    infer.denoiser_model = "denoiser"
    infer.vae_model = _FakeVAE()
    infer.diffusion = SimpleNamespace(
        p_sample_loop=lambda *args, **kwargs: torch.ones((2, 1, 3), dtype=torch.float32),
        ddim_sample_loop=lambda *args, **kwargs: torch.full((2, 1, 3), 2.0, dtype=torch.float32),
    )
    monkeypatch.setattr(inf_mod.clip, "tokenize", lambda texts, truncate=True: torch.ones((len(texts), 3), dtype=torch.long))
    assert torch.allclose(
        infer.normalize(torch.full((1, 1, 5), 5.0)),
        torch.full((1, 1, 5), 2.0),
    )
    assert torch.allclose(
        infer.denormalize(torch.full((1, 1, 5), 2.0)),
        torch.full((1, 1, 5), 5.0),
    )
    embeddings = infer.encode_text(["", "walk"])
    assert torch.count_nonzero(embeddings[0]) == 0
    assert torch.count_nonzero(embeddings[1]) > 0

    infer.respacing = ""
    out = infer.generate_step(
        text_embedding=torch.ones((2, 4), dtype=torch.float32),
        history_motion=torch.ones((2, 2, 5), dtype=torch.float32),
        future_length=4,
        noise=torch.zeros((2, 1, 3), dtype=torch.float32),
    )
    assert out.shape == (2, 4, 5)

    infer.respacing = "ddim2"
    out2 = infer.generate_step(
        text_embedding=torch.ones((2, 4), dtype=torch.float32),
        history_motion=torch.ones((2, 2, 5), dtype=torch.float32),
        future_length=2,
    )
    assert out2.shape == (2, 2, 5)
