"""
DartControl inference engine.

Handles model loading, CLIP text encoding, and autoregressive motion generation.
Pure inference - no training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import clip
import torch
import yaml

from .models.denoiser import DenoiserMLP, DenoiserTransformer, ClassifierFreeWrapper
from .models.vae import AutoMldVae
from .diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)
from .diffusion.respace import SpacedDiffusion, space_timesteps


# ── Config dataclasses (mirroring DART's tyro configs) ──────────────────────


@dataclass
class DiffusionArgs:
    diffusion_steps: int = 10
    respacing: str = ""
    noise_schedule: Literal["linear", "cosine"] = "cosine"
    sigma_small: bool = True


@dataclass
class DenoiserMLPArgs:
    h_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.1
    clip_dim: int = 512
    history_shape: tuple = (2, 276)
    noise_shape: tuple = (1, 128)


@dataclass
class DenoiserTransformerArgs:
    h_dim: int = 512
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.1
    clip_dim: int = 512
    history_shape: tuple = (2, 276)
    noise_shape: tuple = (1, 256)


@dataclass
class DenoiserArgs:
    mvae_path: str = ""
    rescale_latent: int = 1
    train_rollout_type: Literal["single", "full"] = "single"
    train_rollout_history: str = "gt"
    model_type: str = "mlp"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = field(
        default_factory=DenoiserMLPArgs
    )
    diffusion_args: DiffusionArgs = field(default_factory=DiffusionArgs)


@dataclass
class DataArgs:
    cfg_path: str = ""
    data_dir: str = ""
    dataset: str = ""
    prob_static: float = 0.0
    enforce_gender: str = "male"
    enforce_zero_beta: int = 1
    weight_scheme: str = ""
    text_tolerance: float = 0.0
    history_length: int = 0
    future_length: int = 0
    num_primitive: int = 0
    feature_dim: int = 0


@dataclass
class VAEArgs:
    arch: str = "all_encoder"
    ff_size: int = 1024
    num_layers: int = 5
    num_heads: int = 4
    dropout: float = 0.1
    normalize_before: bool = False
    activation: str = "gelu"
    position_embedding: str = "learned"
    latent_dim: tuple[int, int] = (1, 128)
    h_dim: int = 256
    nfeats: int = 0


@dataclass
class MLDArgs:
    """Top-level config for the MLD denoiser checkpoint."""

    data_args: DataArgs = field(default_factory=DataArgs)
    denoiser_args: DenoiserArgs = field(default_factory=DenoiserArgs)
    exp_name: str = "mld"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./mld_denoiser"
    track: int = 1
    wandb_project_name: str = "mld_denoiser"
    wandb_entity: str = "interaction"


@dataclass
class MVAEArgs:
    """Top-level config for the VAE checkpoint."""

    model_args: VAEArgs = field(default_factory=VAEArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    exp_name: str = "mvae"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./mvae"
    track: int = 1
    wandb_project_name: str = "mld_vae"
    wandb_entity: str = "interaction"


# ── Inference engine ────────────────────────────────────────────────────────


def _load_tyro_yaml(path: Path, cls: type):
    """Load a tyro-formatted YAML config file.

    Handles the tyro format where the file is a YAML-encoded string with
    !dataclass tags. Strips tags and parses into a plain dict, then
    manually constructs the dataclass.
    """
    import re

    with open(path, "r") as f:
        text = f.read()
    # Strip all YAML tags (!dataclass:X, !!python/tuple, etc.)
    cleaned = re.sub(r"![!a-zA-Z_:/]+", "", text)
    raw = yaml.safe_load(cleaned) or {}
    # If result is a string, it was double-encoded — parse again
    if isinstance(raw, str):
        cleaned = re.sub(r"![!a-zA-Z_:/]+", "", raw)
        raw = yaml.safe_load(cleaned) or {}
    if not isinstance(raw, dict):
        raw = {}
    # For MLDArgs: extract denoiser_args sub-dict
    if cls is MLDArgs and "denoiser_args" in raw:
        da = raw["denoiser_args"]
        model_args_data = da.get("model_args", {})
        diff_args_data = da.get("diffusion_args", {})
        # Convert tuple fields
        for k in ("history_shape", "noise_shape"):
            if k in model_args_data and isinstance(model_args_data[k], list):
                model_args_data[k] = tuple(model_args_data[k])
        model_type = da.get("model_type", "mlp")
        if "transformer" in model_type.lower():
            model_args = DenoiserTransformerArgs(
                **{
                    k: v
                    for k, v in model_args_data.items()
                    if k in DenoiserTransformerArgs.__dataclass_fields__
                }
            )
        else:
            model_args = DenoiserMLPArgs(
                **{
                    k: v
                    for k, v in model_args_data.items()
                    if k in DenoiserMLPArgs.__dataclass_fields__
                }
            )
        diff_args = DiffusionArgs(
            **{
                k: v
                for k, v in diff_args_data.items()
                if k in DiffusionArgs.__dataclass_fields__
            }
        )
        denoiser_args = DenoiserArgs(
            model_type=model_type,
            model_args=model_args,
            diffusion_args=diff_args,
            mvae_path=da.get("mvae_path", ""),
            rescale_latent=da.get("rescale_latent", 1),
        )
        return MLDArgs(denoiser_args=denoiser_args)
    # For MVAEArgs: extract model_args sub-dict
    if cls is MVAEArgs and "model_args" in raw:
        ma = raw["model_args"]
        if "latent_dim" in ma and isinstance(ma["latent_dim"], list):
            ma["latent_dim"] = tuple(ma["latent_dim"])
        vae_args = VAEArgs(
            **{k: v for k, v in ma.items() if k in VAEArgs.__dataclass_fields__}
        )
        return MVAEArgs(model_args=vae_args)
    return cls()


def _create_diffusion(
    diff_args: DiffusionArgs,
    respacing_override: str = "",
) -> GaussianDiffusion:
    """Create a GaussianDiffusion instance from config."""
    betas = get_named_beta_schedule(diff_args.noise_schedule, diff_args.diffusion_steps)
    respacing = respacing_override or diff_args.respacing

    kwargs = dict(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    if respacing == "" or respacing is None:
        return GaussianDiffusion(**kwargs)
    else:
        use_timesteps = space_timesteps(diff_args.diffusion_steps, respacing)
        return SpacedDiffusion(use_timesteps=use_timesteps, **kwargs)


class DartControlInference:
    """
    DartControl inference engine.

    Loads CLIP, denoiser, and VAE models. Encodes text instructions into
    CLIP embeddings and autoregressively generates motion primitives.
    """

    def __init__(
        self,
        denoiser_checkpoint: str,
        vae_checkpoint: str,
        mean_std_path: str = "assets/dart_control/mean_std_h2_f8.pkl",
        device: str = "cuda",
        respacing: str = "",
        clip_version: str = "ViT-B/32",
    ):
        self.device = device
        self.respacing = respacing

        # Load normalization statistics
        print("[DartControl] Loading normalization statistics...")
        import pickle

        with open(mean_std_path, "rb") as f:
            mean, std = pickle.load(f)
        self._mean = mean.to(device)  # [1, 1, 276]
        self._std = std.to(device)  # [1, 1, 276]

        # Load CLIP for text encoding (OpenAI's clip package, matching DART)
        print("[DartControl] Loading CLIP text encoder...")
        clip_model, _ = clip.load(clip_version, device=device, jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self._clip_model = clip_model

        # Load denoiser + VAE configs and models
        print("[DartControl] Loading denoiser and VAE checkpoints...")
        self.mld_args, self.denoiser_model, self.mvae_args, self.vae_model = (
            self._load_models(denoiser_checkpoint, vae_checkpoint)
        )

        denoiser_args = self.mld_args.denoiser_args

        # Create diffusion sampler
        self.diffusion = _create_diffusion(
            denoiser_args.diffusion_args,
            respacing_override=self.respacing,
        )

        # Key dimensions from checkpoint config
        self.noise_shape = tuple(denoiser_args.model_args.noise_shape)
        self.history_shape = tuple(denoiser_args.model_args.history_shape)
        self.rescale_latent = bool(denoiser_args.rescale_latent)

        print(
            f"[DartControl] Ready. noise_shape={self.noise_shape}, history_shape={self.history_shape}"
        )

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize features from real-world space to model space."""
        return (tensor - self._mean) / self._std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize features from model space to real-world space."""
        return tensor * self._std + self._mean

    def _load_models(self, denoiser_checkpoint: str, vae_checkpoint: str):
        # Load denoiser config and model
        denoiser_dir = Path(denoiser_checkpoint).parent
        mld_args = _load_tyro_yaml(denoiser_dir / "args.yaml", MLDArgs)
        denoiser_args = mld_args.denoiser_args
        model_args = denoiser_args.model_args

        print(f"[DartControl] Denoiser type: {denoiser_args.model_type}")
        if (
            isinstance(model_args, DenoiserTransformerArgs)
            or "transformer" in denoiser_args.model_type.lower()
        ):
            denoiser_class: type[DenoiserTransformer] | type[DenoiserMLP] = (
                DenoiserTransformer
            )
            model_kwargs = {
                "h_dim": model_args.h_dim,
                "ff_size": model_args.ff_size,
                "num_layers": model_args.num_layers,
                "num_heads": model_args.num_heads,
                "dropout": model_args.dropout,
                "activation": model_args.activation,
                "cond_mask_prob": model_args.cond_mask_prob,
                "clip_dim": model_args.clip_dim,
                "history_shape": tuple(model_args.history_shape),
                "noise_shape": tuple(model_args.noise_shape),
            }
        else:
            denoiser_class = DenoiserMLP
            model_kwargs = {
                "h_dim": model_args.h_dim,
                "n_blocks": model_args.n_blocks,
                "dropout": model_args.dropout,
                "activation": model_args.activation,
                "cond_mask_prob": model_args.cond_mask_prob,
                "clip_dim": model_args.clip_dim,
                "history_shape": tuple(model_args.history_shape),
                "noise_shape": tuple(model_args.noise_shape),
            }

        denoiser_model = denoiser_class(**model_kwargs).to(self.device)
        ckpt = torch.load(
            denoiser_checkpoint, map_location=self.device, weights_only=False
        )
        denoiser_model.load_state_dict(ckpt["model_state_dict"])
        denoiser_model.eval()
        for p in denoiser_model.parameters():
            p.requires_grad = False
        wrapped_model = ClassifierFreeWrapper(denoiser_model)

        # Load VAE config and model
        vae_dir = Path(vae_checkpoint).parent
        mvae_args = _load_tyro_yaml(vae_dir / "args.yaml", MVAEArgs)
        vae_model_args = mvae_args.model_args

        print(f"[DartControl] VAE arch: {vae_model_args.arch}")
        vae_model = AutoMldVae(
            arch=vae_model_args.arch,
            ff_size=vae_model_args.ff_size,
            num_layers=vae_model_args.num_layers,
            num_heads=vae_model_args.num_heads,
            dropout=vae_model_args.dropout,
            normalize_before=vae_model_args.normalize_before,
            activation=vae_model_args.activation,
            position_embedding=vae_model_args.position_embedding,
            latent_dim=tuple(vae_model_args.latent_dim),
            h_dim=vae_model_args.h_dim,
            nfeats=vae_model_args.nfeats,
        ).to(self.device)
        vae_ckpt = torch.load(
            vae_checkpoint, map_location=self.device, weights_only=False
        )
        vae_state = vae_ckpt["model_state_dict"]
        if "latent_mean" not in vae_state:
            vae_state["latent_mean"] = torch.tensor(0)
        if "latent_std" not in vae_state:
            vae_state["latent_std"] = torch.tensor(1)
        vae_model.load_state_dict(vae_state)
        vae_model.latent_mean = vae_state["latent_mean"]
        vae_model.latent_std = vae_state["latent_std"]
        vae_model.eval()
        for p in vae_model.parameters():
            p.requires_grad = False

        return mld_args, wrapped_model, mvae_args, vae_model

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text instructions into CLIP embeddings using OpenAI's clip package.

        Args:
            texts: List of instruction strings (e.g. ["walk", "sit", "wave"])

        Returns:
            embeddings: [len(texts), 512] float32 tensor
        """
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            embeddings = self._clip_model.encode_text(tokens).float()
        # Zero out empty strings (matches original DART behavior)
        for i, t in enumerate(texts):
            if t == "":
                embeddings[i] = 0.0
        return embeddings

    @torch.no_grad()
    def generate_step(
        self,
        text_embedding: torch.Tensor,
        history_motion: torch.Tensor,
        guidance_scale: float = 5.0,
        future_length: int = 8,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate one motion primitive (one autoregressive step).

        Args:
            text_embedding: [B, 512] CLIP text embedding
            history_motion: [B, H, nfeats] normalized history
            guidance_scale: Classifier-free guidance strength
            future_length: Frames to generate per step
            noise: Optional [B, *noise_shape] noise tensor from RL policy.
                   None means random noise (default behavior).

        Returns:
            future_motion: [B, F, nfeats] normalized future features
        """
        batch_size = text_embedding.shape[0]
        guidance = (
            torch.ones(batch_size, *self.noise_shape, device=self.device)
            * guidance_scale
        )

        y = {
            "text_embedding": text_embedding,
            "history_motion_normalized": history_motion,
            "scale": guidance,
        }

        sample_fn = (
            self.diffusion.ddim_sample_loop
            if self.respacing
            else self.diffusion.p_sample_loop
        )
        x_start = sample_fn(
            self.denoiser_model,
            (batch_size, *self.noise_shape),
            clip_denoised=False,
            model_kwargs={"y": y},
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )  # [B, T=1, D]

        latent = x_start.permute(1, 0, 2)  # [T=1, B, D]
        future_motion = self.vae_model.decode(
            latent,
            history_motion,
            nfuture=future_length,
            scale_latent=self.rescale_latent,
        )  # [B, F, nfeats]

        return future_motion
