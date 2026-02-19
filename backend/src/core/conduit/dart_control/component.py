"""
DartControl conduit component.

Takes text instructions (e.g. "walk", "sit", "wave") as input and outputs
BodyPoseFrames with position + quaternion for 13 tracked body parts.
"""

from __future__ import annotations

import math
import random
import threading
import time
from queue import Empty, Queue
from typing import TypedDict

import torch
from pydantic import BaseModel

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import BodyPoseFrame, BonePose

from .inference import DartControlInference

STUB_INSTRUCTIONS = ["walk", "sit", "wave", "stand up", "turn around", "jump"]


# ── SMPL feature layout (276 dims) ─────────────────────────────────────────
# transl:                  0:3     (3)
# poses_6d:                3:135   (22 joints × 6)
# transl_delta:            135:138 (3)
# global_orient_delta_6d:  138:144 (6)
# joints:                  144:210 (22 joints × 3)
# joints_delta:            210:276 (66)

_JOINTS_OFFSET = 144
_POSES_6D_OFFSET = 3
_N_JOINTS = 22

# SMPL joint index → OpenVR body part name
_SMPL_TO_OPENVR: dict[int, str] = {
    15: "head",
    20: "left_hand",
    21: "right_hand",
    0:  "waist",
    9:  "chest",
    10: "left_foot",
    11: "right_foot",
    4:  "left_knee",
    5:  "right_knee",
    18: "left_elbow",
    19: "right_elbow",
    16: "left_shoulder",
    17: "right_shoulder",
}


def _rot6d_to_quaternion(rot6d: torch.Tensor) -> tuple[float, float, float, float]:
    """Convert a 6D continuous rotation to quaternion (w, x, y, z).

    The 6D representation is the first two columns of the rotation matrix.
    Uses Gram-Schmidt to reconstruct the full rotation matrix, then converts.
    """
    a1 = rot6d[:3]
    a2 = rot6d[3:6]

    # Gram-Schmidt: orthonormalize
    e1 = a1 / (a1.norm() + 1e-8)
    e2 = a2 - (e2_proj := (a2 @ e1) * e1)
    e2 = e2 / (e2.norm() + 1e-8)
    e3 = torch.linalg.cross(e1, e2)

    # Rotation matrix columns → rows of 3×3
    m00, m01, m02 = e1[0].item(), e2[0].item(), e3[0].item()
    m10, m11, m12 = e1[1].item(), e2[1].item(), e3[1].item()
    m20, m21, m22 = e1[2].item(), e2[2].item(), e3[2].item()

    # Rotation matrix to quaternion (Shepperd's method)
    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return (w, x, y, z)


def _features_to_body_pose(features: torch.Tensor) -> dict[str, BonePose | None]:
    """Convert a single frame of 276 SMPL features to a dict of BonePoses.

    Args:
        features: [276] tensor — one frame of motion features.

    Returns:
        Dict mapping OpenVR body part names to BonePose.
    """
    joints = features[_JOINTS_OFFSET:_JOINTS_OFFSET + _N_JOINTS * 3].reshape(_N_JOINTS, 3)
    poses_6d = features[_POSES_6D_OFFSET:_POSES_6D_OFFSET + _N_JOINTS * 6].reshape(_N_JOINTS, 6)

    poses: dict[str, BonePose | None] = {}
    for joint_idx, part_name in _SMPL_TO_OPENVR.items():
        pos = joints[joint_idx]
        qw, qx, qy, qz = _rot6d_to_quaternion(poses_6d[joint_idx])
        poses[part_name] = BonePose(
            pos_x=pos[0].item(), pos_y=pos[1].item(), pos_z=pos[2].item(),
            rot_w=qw, rot_x=qx, rot_y=qy, rot_z=qz,
        )
    return poses


class DartControlConfig(BaseModel):
    denoiser_checkpoint: str = "assets/dart_control/mld/checkpoint_300000.pt"
    """Path to the denoiser .pt checkpoint file."""

    vae_checkpoint: str = "assets/dart_control/mvae/checkpoint_200000.pt"
    """Path to the VAE .pt checkpoint file."""

    device: str = "cuda"
    """Device for inference (cuda or cpu)."""

    respacing: str = ""
    """DDIM respacing (e.g. 'ddim10'). Empty string for full diffusion sampling."""

    clip_model_name: str = "openai/clip-vit-base-patch32"
    """HuggingFace CLIP model for text encoding."""

    guidance_scale: float = 5.0
    """Classifier-free guidance strength."""

    future_length: int = 8
    """Number of motion frames generated per primitive step."""

    batch_size: int = 1
    """Batch size for motion generation."""

    stub_interval: float = 3.0
    """Seconds between switching to a new random instruction."""

    fps: float = 30.0
    """Framerate for emitting body pose frames."""


class DartControlOutputs(TypedDict):
    motion: Channel[BodyPoseFrame]


class DartControl(Component[[], DartControlOutputs]):
    """
    DartControl motion generation conduit.

    Generates body poses from stub text instructions (walk, sit, wave, etc.)
    and emits BodyPoseFrames on the output channel.
    """

    def __init__(self, config: DartControlConfig) -> None:
        super().__init__()
        self.config = config
        self._output_motion = Channel[BodyPoseFrame](name="motion")

        self._engine: DartControlInference | None = None
        self._history: torch.Tensor | None = None

    def get_output_channels(self) -> DartControlOutputs:
        return {"motion": self._output_motion}

    def _ensure_engine(self) -> DartControlInference:
        """Lazy-load the inference engine on first use."""
        if self._engine is None:
            self._engine = DartControlInference(
                denoiser_checkpoint=self.config.denoiser_checkpoint,
                vae_checkpoint=self.config.vae_checkpoint,
                device=self.config.device,
                respacing=self.config.respacing,
                clip_model_name=self.config.clip_model_name,
            )
        return self._engine

    def _ensure_history(self, engine: DartControlInference) -> torch.Tensor:
        """Initialize history motion with zeros if not set."""
        if self._history is None:
            h_len, nfeats = engine.history_shape
            self._history = torch.zeros(
                self.config.batch_size, h_len, nfeats,
                device=self.config.device, dtype=torch.float32
            )
        return self._history

    def _generate_primitive(self, engine: DartControlInference, instruction: str) -> list[dict[str, BonePose | None]]:
        """Generate one motion primitive and return converted body poses."""
        history = self._ensure_history(engine)

        text_embedding = engine.encode_text([instruction])  # [1, 512]
        text_embedding = text_embedding.expand(self.config.batch_size, -1)

        result = engine.generate_step(
            text_embedding=text_embedding,
            history_motion=history,
            guidance_scale=self.config.guidance_scale,
            future_length=self.config.future_length,
        )

        self._history = result.history

        features = result.features.cpu()  # [B, F, 276]
        return [_features_to_body_pose(features[0, f]) for f in range(features.shape[1])]

    def run(self) -> None:
        print("[DartControl] Starting DartControl component, loading models...")
        engine = self._ensure_engine()
        print("[DartControl] Models loaded, streaming motion")

        frame_interval = 1.0 / self.config.fps
        instruction = random.choice(STUB_INSTRUCTIONS)
        last_switch = time.monotonic()
        print(f"[DartControl] Instruction: '{instruction}'")

        while not self.stop_event.is_set():
            # Switch instruction periodically
            now = time.monotonic()
            if now - last_switch >= self.config.stub_interval:
                instruction = random.choice(STUB_INSTRUCTIONS)
                last_switch = now
                print(f"[DartControl] Instruction: '{instruction}'")

            # Generate a batch of frames
            try:
                pose_batch = self._generate_primitive(engine, instruction)
            except Exception as e:
                print(f"[DartControl] Generation error for '{instruction}': {e}")
                continue

            # Emit each frame at the target framerate
            for body_poses in pose_batch:
                if self.stop_event.is_set():
                    break
                send_time = time.monotonic()
                self._output_motion.send(BodyPoseFrame(poses=body_poses))
                elapsed = time.monotonic() - send_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    self.stop_event.wait(sleep_time)

        print("[DartControl] Component stopped")
