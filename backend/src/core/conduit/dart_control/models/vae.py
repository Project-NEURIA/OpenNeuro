"""
Motion VAE model for DartControl.
Compresses motion sequences into latent space.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .operators import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    build_position_encoding,
)


class AutoMldVae(nn.Module):
    """
    Autoregressive Motion Latent Diffusion VAE.

    Encodes motion sequences into latent space and decodes them back.
    Uses skip-connection transformer architecture for both encoder and decoder.
    """

    def __init__(
        self,
        nfeats: int,
        latent_dim: tuple = (1, 256),
        h_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "all_encoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        **kwargs,
    ) -> None:
        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False
        self.pe_type = "mld"

        # Position encodings
        self.query_pos_encoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding
        )
        self.query_pos_decoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding
        )

        # Encoder
        encoder_layer = TransformerEncoderLayer(
            self.h_dim, num_heads, ff_size, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        # Decoder
        self.decoder: SkipTransformerEncoder | SkipTransformerDecoder
        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(
                encoder_layer, num_layers, decoder_norm
            )
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim, num_heads, ff_size, dropout, activation, normalize_before
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(
                decoder_layer, num_layers, decoder_norm
            )
        else:
            raise ValueError("Not support architecture!")

        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)

        # Embeddings
        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.h_dim)
        )
        self.skel_embedding = nn.Linear(input_feats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, output_feats)

        # Normalization buffers
        self.register_buffer("latent_mean", torch.tensor(0))
        self.register_buffer("latent_std", torch.tensor(1))

    def encode(
        self,
        future_motion: Tensor,
        history_motion: Tensor,
        scale_latent: bool = False,
    ) -> tuple[Tensor, Any]:
        """
        Encode motion into latent space.

        Args:
            future_motion: [B, F, nfeats] - Future motion frames
            history_motion: [B, H, nfeats] - History motion frames
            scale_latent: Whether to scale latent by learned std

        Returns:
            latent: [latent_size, B, latent_dim] - Sampled latent
            dist: Normal distribution for KL loss
        """
        bs, nfuture, nfeats = future_motion.shape

        x = torch.cat((history_motion, future_motion), dim=1)  # [B, H+F, nfeats]
        x = self.skel_embedding(x)
        x = x.permute(1, 0, 2)  # [nframes, B, h_dim]

        # Add global motion tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq)[: dist.shape[0]]  # [2*latent_size, B, h_dim]
        dist = self.encoder_latent_proj(dist)  # [2*latent_size, B, latent_dim]

        # Split into mean and logvar
        mu = dist[0 : self.latent_size, ...]
        logvar = dist[self.latent_size :, ...]
        logvar = torch.clamp(logvar, min=-10, max=10)

        # Reparameterization
        std = logvar.exp().pow(0.5)
        normal = torch.distributions.Normal(mu, std)
        latent = normal.rsample()

        if scale_latent:
            latent = latent / self.latent_std  # type: ignore[operator]

        return latent, normal

    def decode(
        self,
        z: Tensor,
        history_motion: Tensor,
        nfuture: int,
        scale_latent: bool = False,
    ) -> Tensor:
        """
        Decode latent into motion.

        Args:
            z: [latent_size, B, latent_dim] - Latent vector
            history_motion: [B, H, nfeats] - History motion frames
            nfuture: Number of future frames to generate
            scale_latent: Whether to unscale latent by learned std

        Returns:
            feats: [B, F, nfeats] - Generated future motion
        """
        bs = history_motion.shape[0]

        if scale_latent:
            z = z * self.latent_std  # type: ignore[operator]

        z = self.decoder_latent_proj(z)  # [latent_size, B, h_dim]
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device)
        history_embedding = self.skel_embedding(history_motion).permute(
            1, 0, 2
        )  # [H, B, h_dim]

        if self.arch == "all_encoder":
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[-nfuture:]
        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(tgt=xseq, memory=z)
            output = output[-nfuture:]

        output = self.final_layer(output)
        feats = output.permute(1, 0, 2)  # [B, F, nfeats]
        return feats
