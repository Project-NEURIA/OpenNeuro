"""
Denoiser models for DartControl.
Generates motion latents conditioned on text (CLIP embeddings) and history motion.
"""

import numpy as np
import torch
import torch.nn as nn

from .operators import PositionalEncoding


class TimestepEmbedder(nn.Module):
    """Embeds diffusion timesteps into a learned representation."""

    def __init__(self, h_dim, sequence_pos_encoder):
        super().__init__()
        self.h_dim = h_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        self.time_embed = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class MLP(nn.Module):
    """Simple MLP with configurable activation."""

    def __init__(self, in_dim, h_dims=(128, 128), activation="tanh"):
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
            self.activation = torch.tanh

        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_dim_, h_dim))
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class MLPBlock(nn.Module):
    """Residual MLP block."""

    def __init__(self, h_dim, out_dim, n_blocks, actfun="relu", residual=True):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            [
                MLP(h_dim, h_dims=(h_dim, h_dim), activation=actfun)
                for _ in range(n_blocks)
            ]
        )
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        y = self.out_fc(h)
        return y


class DenoiserMLP(nn.Module):
    """
    MLP-based denoiser for motion diffusion.

    Takes noisy latent, timestep, text embedding, and history motion as input.
    Outputs denoised latent prediction.
    """

    def __init__(
        self,
        h_dim: int = 512,
        n_blocks: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        clip_dim: int = 512,
        history_shape: tuple = (2, 276),
        noise_shape: tuple = (1, 128),
        cond_mask_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation
        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.clip_dim = clip_dim
        self.cond_mask_prob = cond_mask_prob

        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim, self.sequence_pos_encoder)

        input_dim = (
            self.h_dim + self.clip_dim + np.prod(history_shape) + np.prod(noise_shape)
        )
        self.input_project = nn.Linear(input_dim, self.h_dim)

        self.mlp = MLPBlock(
            h_dim=h_dim,
            out_dim=np.prod(noise_shape),
            n_blocks=n_blocks,
            actfun=activation,
        )

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x_t, timesteps, y=None):
        """
        Forward pass.

        Args:
            x_t: [B, T=1, D] - Noisy latent
            timesteps: [B] - Diffusion timesteps
            y: dict with 'text_embedding', 'history_motion_normalized', 'uncond'

        Returns:
            output: [B, T=1, D] - Predicted clean latent
        """
        batch_size = x_t.shape[0]

        emb_time = self.embed_timestep(timesteps).squeeze(0)  # [B, h_dim]
        emb_history = y["history_motion_normalized"].reshape(
            batch_size, np.prod(self.history_shape)
        )
        force_mask = y.get("uncond", False)
        emb_text = self.mask_cond(y["text_embedding"], force_mask=force_mask)
        emb_noise = x_t.reshape(batch_size, np.prod(self.noise_shape))

        input_embed = torch.cat((emb_time, emb_text, emb_history, emb_noise), dim=1)
        output = self.mlp(self.input_project(input_embed))
        output = output.reshape(batch_size, *self.noise_shape)

        return output


class DenoiserTransformer(nn.Module):
    """
    Transformer-based denoiser for motion diffusion.

    Takes noisy latent, timestep, text embedding, and history motion as input.
    Uses transformer encoder to process the sequence.
    """

    def __init__(
        self,
        h_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        clip_dim: int = 512,
        history_shape: tuple = (2, 276),
        noise_shape: tuple = (1, 128),
        cond_mask_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.clip_dim = clip_dim
        self.cond_mask_prob = cond_mask_prob

        # Input embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.h_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Output projection
        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x_t, timesteps, y=None):
        """
        Forward pass.

        Args:
            x_t: [B, T=1, D] - Noisy latent
            timesteps: [B] - Diffusion timesteps
            y: dict with 'text_embedding', 'history_motion_normalized', 'uncond'

        Returns:
            output: [B, T=1, D] - Predicted clean latent
        """
        emb_time = self.embed_timestep(timesteps)  # [1, B, h_dim]
        emb_history = self.embed_history(y["history_motion_normalized"]).permute(
            1, 0, 2
        )  # [H, B, h_dim]
        force_mask = y.get("uncond", False)
        emb_text = self.embed_text(
            self.mask_cond(y["text_embedding"], force_mask=force_mask)
        ).unsqueeze(0)  # [1, B, h_dim]
        emb_noise = self.embed_noise(x_t).permute(1, 0, 2)  # [1, B, h_dim]

        xseq = torch.cat((emb_time, emb_text, emb_history, emb_noise), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[-self.noise_shape[0] :]  # [1, B, h_dim]
        output = self.output_process(output)  # [1, B, noise_shape[-1]]
        output = output.permute(1, 0, 2)  # [B, 1, noise_shape[-1]]

        return output


class ClassifierFreeWrapper(nn.Module):
    """
    Wrapper for classifier-free guidance during sampling.

    Runs the model twice (conditional and unconditional) and
    interpolates based on guidance scale.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        assert self.model.cond_mask_prob > 0, (
            "Cannot run guided diffusion on a model not trained with condition masking"
        )

    def forward(self, x, timesteps, y=None):
        """
        Forward with classifier-free guidance.

        Args:
            x: Noisy latent
            timesteps: Diffusion timesteps
            y: dict with 'scale' for guidance strength

        Returns:
            Guided output = uncond + scale * (cond - uncond)
        """
        y["uncond"] = False
        out = self.model(x, timesteps, y)

        y["uncond"] = True
        out_uncond = self.model(x, timesteps, y)

        return out_uncond + (y["scale"] * (out - out_uncond))
