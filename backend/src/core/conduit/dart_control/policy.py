"""
RL policy for goal-reaching control.

Ported from DART's control/policy/policy.py with minimal changes:
- Replaced loralib.Linear with nn.Linear (config has use_lora=0)
- Stripped critic network (inference only)
- Added get_action_mean() for deterministic action output
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    activation: Any

    def __init__(
        self,
        in_dim: int,
        h_dims: list[int] = [128, 128],
        activation: str = "tanh",
    ):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_dim_, h_dim))
            in_dim_ = h_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        h_dim: int,
        out_dim: int,
        n_blocks: int,
        actfun: str = "relu",
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            [MLP(h_dim, h_dims=[h_dim, h_dim], activation=actfun) for _ in range(n_blocks)]
        )
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        return self.out_fc(h)


@dataclass
class PolicyConfig:
    """Configuration matching DART's RL policy for reach-location task."""

    latent_dim: int = 512
    n_blocks: int = 2
    activation: str = "lrelu"
    pred_std: bool = False
    use_tanh_scale: bool = True
    # Observation component dimensions
    motion_dim: int = 552       # history_length(2) * feature_dim(276)
    text_dim: int = 512         # CLIP embedding dim
    goal_dim: int = 4           # goal_dir(3) + goal_dist(1)
    scene_dim: int = 1          # floor height
    # Action dimension = product of noise_shape
    action_dim: int = 256       # 1 * 256 for transformer denoiser


class PolicyReachLocationMLP(nn.Module):
    """
    RL policy that maps (goal, motion history, text, scene) observations
    to a noise tensor for the diffusion sampler.

    Inference-only version: no critic, no lora.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim
        action_dim = config.action_dim
        activation = config.activation

        # Observation structure indices (same order as env_reach_location_mld.py)
        # goal_dir(3), goal_dist(1), text(512), motion(552), scene(1)
        self._obs_slices = self._build_obs_slices(config)

        self.motion_encoder = MLP(in_dim=config.motion_dim, h_dims=[latent_dim], activation=activation)
        self.text_encoder = MLP(in_dim=config.text_dim, h_dims=[latent_dim], activation=activation)
        self.goal_encoder = MLP(in_dim=config.goal_dim, h_dims=[latent_dim], activation=activation)
        self.scene_encoder = MLP(in_dim=config.scene_dim, h_dims=[latent_dim], activation=activation)
        self.embedding_encoder = MLP(in_dim=latent_dim * 4, h_dims=[latent_dim], activation=activation)

        out_dim = action_dim * 2 if config.pred_std else action_dim
        self.actor = MLPBlock(latent_dim, out_dim, config.n_blocks, actfun=activation)

        if not config.pred_std:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    @staticmethod
    def _build_obs_slices(config: PolicyConfig) -> dict[str, tuple[int, int]]:
        """Compute start/end indices for each observation component."""
        slices = {}
        idx = 0
        for name, dim in [
            ("goal_dir", 3),
            ("goal_dist", 1),
            ("text", config.text_dim),
            ("motion", config.motion_dim),
            ("scene", config.scene_dim),
        ]:
            slices[name] = (idx, idx + dim)
            idx += dim
        return slices

    def get_embedding(self, observation: torch.Tensor) -> torch.Tensor:
        s = self._obs_slices
        motion = observation[:, s["motion"][0]:s["motion"][1]]
        text = observation[:, s["text"][0]:s["text"][1]]
        goal_dir = observation[:, s["goal_dir"][0]:s["goal_dir"][1]]
        goal_dist = observation[:, s["goal_dist"][0]:s["goal_dist"][1]]
        goal = torch.cat((goal_dir, goal_dist), dim=1)
        scene = observation[:, s["scene"][0]:s["scene"][1]]

        motion_emb = self.motion_encoder(motion)
        text_emb = self.text_encoder(text)
        goal_emb = self.goal_encoder(goal)
        scene_emb = self.scene_encoder(scene)

        embedding = torch.cat((motion_emb, text_emb, goal_emb, scene_emb), dim=1)
        return self.embedding_encoder(embedding)

    @torch.no_grad()
    def get_action_mean(self, observation: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass: returns the mean action (no sampling).

        Args:
            observation: [B, obs_dim] observation vector

        Returns:
            action_mean: [B, action_dim] noise tensor for diffusion sampler
        """
        embedding = self.get_embedding(observation)
        actor_output = self.actor(embedding)
        action_mean = actor_output[:, :self.config.action_dim]
        if self.config.use_tanh_scale:
            action_mean = torch.tanh(action_mean) * 4
        return action_mean
