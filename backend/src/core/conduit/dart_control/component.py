"""
DartControl conduit component.

Takes text instructions (e.g. "walk", "sit", "wave") and optional goal
coordinates as input and outputs BodyPoseFrames with position + quaternion
for 13 tracked body parts.
"""

from __future__ import annotations

import math
import pickle
import queue
import threading
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import torch
from . import rotation_conversions as transforms
from pydantic import BaseModel

from src.core.component import ThreadedComponent, Tag
from src.core.channel import Receiver, Sender
from src.core.frames import BodyPoseFrame, BonePose, GoalFrame, TextFrame

from .inference import DartControlInference
from .policy import PolicyConfig, PolicyReachLocationMLP
from .smpl_utils import PrimitiveUtility, get_new_coordinate


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
    0: "waist",
    9: "chest",
    10: "left_foot",
    11: "right_foot",
    4: "left_knee",
    5: "right_knee",
    18: "left_elbow",
    19: "right_elbow",
    16: "left_shoulder",
    17: "right_shoulder",
}


# Quaternion for -90° rotation around X: converts Z-up (SMPL) → Y-up (standard)
_SQRT2_2 = math.sqrt(2.0) / 2.0
_Q_ZUP_TO_YUP = (_SQRT2_2, -_SQRT2_2, 0.0, 0.0)  # (w, x, y, z)
_Q_ZUP_TO_YUP_INV = (_SQRT2_2, _SQRT2_2, 0.0, 0.0)  # conjugate


def _quat_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Multiply two quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


# SMPLX kinematic tree: parent index for each of the 22 joints
_SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
]


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to 3x3 matrix via Gram-Schmidt."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    e1 = a1 / (a1.norm() + 1e-8)
    e2 = a2 - (a2 @ e1) * e1
    e2 = e2 / (e2.norm() + 1e-8)
    e3 = torch.linalg.cross(e1, e2)
    return torch.stack([e1, e2, e3], dim=0)  # [3, 3] — rows, matching DART


def _rotmat_to_quaternion(m: torch.Tensor) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    m00, m01, m02 = m[0, 0].item(), m[0, 1].item(), m[0, 2].item()
    m10, m11, m12 = m[1, 0].item(), m[1, 1].item(), m[1, 2].item()
    m20, m21, m22 = m[2, 0].item(), m[2, 1].item(), m[2, 2].item()
    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return (0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s)
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)


def _features_to_body_pose(features: torch.Tensor) -> dict[str, BonePose | None]:
    """Convert a single frame of 276 SMPL features to a dict of BonePoses.

    Uses forward kinematics to compute world-space rotations from local
    joint rotations, then converts from SMPL's Z-up to Y-up:
      position: (x, y, z)_zup → (x, z, -y)_yup
      rotation: q_yup = q_conversion * q_zup_world

    Args:
        features: [276] tensor — one frame of denormalized motion features.

    Returns:
        Dict mapping body part names to BonePose in Y-up coordinates.
    """
    joints = features[_JOINTS_OFFSET : _JOINTS_OFFSET + _N_JOINTS * 3].reshape(
        _N_JOINTS, 3
    )
    poses_6d = features[_POSES_6D_OFFSET : _POSES_6D_OFFSET + _N_JOINTS * 6].reshape(
        _N_JOINTS, 6
    )

    # Convert all local rotations to 3x3 matrices
    local_rots = [_rot6d_to_matrix(poses_6d[j]) for j in range(_N_JOINTS)]

    # Forward kinematics: accumulate world rotations through kinematic chain
    world_rots: list[torch.Tensor] = [torch.empty(0)] * _N_JOINTS
    for j in range(_N_JOINTS):
        parent = _SMPL_PARENTS[j]
        if parent == -1:
            world_rots[j] = local_rots[j]  # root = global orient (already world-space)
        else:
            world_rots[j] = world_rots[parent] @ local_rots[j]

    poses: dict[str, BonePose | None] = {}
    for joint_idx, part_name in _SMPL_TO_OPENVR.items():
        pos = joints[joint_idx]
        # World-space quaternion in Z-up
        q_zup = _rotmat_to_quaternion(world_rots[joint_idx])
        # Convert to Y-up
        q_yup = _quat_multiply(_Q_ZUP_TO_YUP, q_zup)
        poses[part_name] = BonePose(
            pos_x=pos[0].item(),
            pos_y=pos[2].item(),
            pos_z=-pos[1].item(),
            rot_w=q_yup[0],
            rot_x=q_yup[1],
            rot_y=q_yup[2],
            rot_z=q_yup[3],
        )
    return poses


class DartControlConfig(BaseModel):
    denoiser_checkpoint: str = "assets/dart_control/mld/checkpoint_300000.pt"
    """Path to the denoiser .pt checkpoint file."""

    vae_checkpoint: str = "assets/dart_control/mvae/checkpoint_200000.pt"
    """Path to the VAE .pt checkpoint file."""

    mean_std_path: str = "assets/dart_control/mean_std_h2_f8.pkl"
    """Path to the normalization statistics pickle file."""

    stand_path: str = "assets/dart_control/stand.pkl"
    """Path to the standing pose pickle file for history initialization."""

    device: Literal["cuda", "cpu", "mps"]
    """Device for inference."""

    respacing: str = ""
    """DDIM respacing (e.g. 'ddim10'). Empty string for full diffusion sampling."""

    clip_version: str = "ViT-B/32"
    """OpenAI CLIP model version for text encoding."""

    guidance_scale: float = 5.0
    """Classifier-free guidance strength."""

    future_length: int = 8
    """Number of motion frames generated per primitive step."""

    batch_size: int = 1
    """Batch size for motion generation."""

    gender: str = "male"
    """SMPL body gender."""

    fps: float = 30.0
    """Framerate for emitting body pose frames."""

    policy_checkpoint: str = "assets/dart_control/iter_2000.pth"
    """Path to trained RL policy checkpoint. Empty string disables policy."""

    obs_goal_angle_clip: float = 60.0
    """Maximum angle (degrees) between body forward dir and goal dir in observation."""

    obs_goal_dist_clip: float = 5.0
    """Maximum goal distance value in observation (clamped)."""


class DartControlInputs(NamedTuple):
    goal: Receiver[GoalFrame] | None = None
    instruction: Receiver[TextFrame] | None = None


class DartControlOutputs(NamedTuple):
    motion: Sender[BodyPoseFrame]


class DartControl(ThreadedComponent[DartControlInputs, DartControlOutputs]):
    description = "Controls DART robot movements from pose data"

    """
    DartControl motion generation conduit.

    Generates body poses from text instructions and emits BodyPoseFrames
    on the output channel. Uses DART's canonicalization pipeline for
    smooth, continuous motion generation.
    """

    tags = Tag(io={"conduit"}, functionality={"movement"})

    def __init__(self, config: DartControlConfig) -> None:
        super().__init__()
        self.config = config
        self._engine: DartControlInference | None = None
        self._primitive_util: PrimitiveUtility | None = None

        # Autoregressive state (set by _init_from_stand before use)
        self._history: torch.Tensor = torch.empty(0)
        self._transf_rotmat: torch.Tensor = torch.empty(0)
        self._transf_transl: torch.Tensor = torch.empty(0)
        self._pelvis_delta: torch.Tensor = torch.empty(0)
        self._betas: torch.Tensor = torch.empty(0)

    def _ensure_engine(self) -> DartControlInference:
        """Lazy-load the inference engine on first use."""
        if self._engine is None:
            self._engine = DartControlInference(
                denoiser_checkpoint=self.config.denoiser_checkpoint,
                vae_checkpoint=self.config.vae_checkpoint,
                mean_std_path=self.config.mean_std_path,
                device=self.config.device,
                respacing=self.config.respacing,
                clip_version=self.config.clip_version,
            )
        return self._engine

    def _ensure_primitive_util(self) -> PrimitiveUtility:
        """Lazy-load the PrimitiveUtility (SMPL body model)."""
        if self._primitive_util is None:
            self._primitive_util = PrimitiveUtility(device=self.config.device)
        return self._primitive_util

    def _init_from_stand(
        self, engine: DartControlInference, putil: PrimitiveUtility
    ) -> None:
        """Initialize history from stand.pkl using DART's canonicalization pipeline."""
        device = self.config.device
        B = self.config.batch_size
        h_len = engine.history_shape[0]  # typically 2
        gender = self.config.gender

        # Load stand.pkl
        with open(self.config.stand_path, "rb") as f:
            stand_data = pickle.load(f)

        # Build primitive dict from stand.pkl (matching DART's get_primitive)
        seq_len = h_len + 1  # need h_len+1 frames to compute h_len delta features
        transl = torch.tensor(
            stand_data["transl"][:seq_len], dtype=torch.float32
        )  # [T, 3]
        global_orient = torch.tensor(
            stand_data["global_orient"][:seq_len], dtype=torch.float32
        )  # [T, 3]
        body_pose = torch.tensor(
            stand_data["body_pose"][:seq_len], dtype=torch.float32
        )  # [T, 63]

        # Pad if stand.pkl has fewer frames than needed
        if transl.shape[0] < seq_len:
            pad = seq_len - transl.shape[0]
            transl = torch.cat([transl, transl[-1:].expand(pad, -1)], dim=0)
            global_orient = torch.cat(
                [global_orient, global_orient[-1:].expand(pad, -1)], dim=0
            )
            body_pose = torch.cat([body_pose, body_pose[-1:].expand(pad, -1)], dim=0)

        # Convert axis-angle to rotation matrices
        global_orient_mat = transforms.axis_angle_to_matrix(global_orient)  # [T, 3, 3]
        body_pose_mat = transforms.axis_angle_to_matrix(
            body_pose.reshape(-1, 3)
        ).reshape(seq_len, 21, 3, 3)  # [T, 21, 3, 3]

        betas = torch.zeros(10, dtype=torch.float32)  # enforce zero betas like DART

        primitive_dict = {
            "gender": gender,
            "betas": betas.unsqueeze(0)
            .unsqueeze(0)
            .expand(B, seq_len, 10)
            .to(device),  # [B, T, 10]
            "transl": transl.unsqueeze(0).expand(B, -1, -1).to(device),  # [B, T, 3]
            "global_orient": global_orient_mat.unsqueeze(0)
            .expand(B, -1, -1, -1)
            .to(device),  # [B, T, 3, 3]
            "body_pose": body_pose_mat.unsqueeze(0)
            .expand(B, -1, -1, -1, -1)
            .to(device),  # [B, T, 21, 3, 3]
            "transf_rotmat": torch.eye(3, device=device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone(),  # [B, 3, 3]
            "transf_transl": torch.zeros(B, 1, 3, device=device),  # [B, 1, 3]
        }

        # Compute pelvis offset (cached for all future steps)
        self._pelvis_delta = putil.calc_calibrate_offset(
            {
                "gender": gender,
                "betas": betas.unsqueeze(0).expand(B, -1).to(device),
            }
        )  # [B, 3]
        primitive_dict["pelvis_delta"] = self._pelvis_delta

        # Canonicalize
        _, _, primitive_dict = putil.canonicalize(primitive_dict)

        # Compute features via SMPL forward kinematics
        feature_dict = putil.calc_features(primitive_dict)

        # Convert to tensor [B, T, 276] — note: calc_features produces T-1 frames for delta fields
        # We need to pad delta features to match T frames
        # The last frame's delta is just repeated from the second-to-last
        feature_dict["transl_delta"] = torch.cat(
            [
                feature_dict["transl_delta"],
                feature_dict["transl_delta"][:, -1:, :],
            ],
            dim=1,
        )
        feature_dict["joints_delta"] = torch.cat(
            [
                feature_dict["joints_delta"],
                feature_dict["joints_delta"][:, -1:, :],
            ],
            dim=1,
        )
        feature_dict["global_orient_delta_6d"] = torch.cat(
            [
                feature_dict["global_orient_delta_6d"],
                feature_dict["global_orient_delta_6d"][:, -1:, :],
            ],
            dim=1,
        )

        history_tensor = putil.dict_to_tensor(feature_dict)  # [B, T, 276]
        # Take last h_len frames
        history_tensor = history_tensor[:, -h_len:, :]  # [B, H, 276]

        # Normalize
        self._history = engine.normalize(history_tensor)
        self._transf_rotmat = primitive_dict["transf_rotmat"]
        self._transf_transl = primitive_dict["transf_transl"]
        self._betas = betas.unsqueeze(0).expand(B, -1).to(device)

        print(
            f"[DartControl] History initialized from stand.pkl (shape={self._history.shape})"
        )

    def _update_history(
        self,
        engine: DartControlInference,
        putil: PrimitiveUtility,
        history_normalized: torch.Tensor,
        future_normalized: torch.Tensor,
    ) -> None:
        """Update history using DART's canonicalization pipeline.

        1. Denormalize [history + future]
        2. Take last history_len frames
        3. Convert to feature dict, attach metadata
        4. get_blended_feature (canonicalize + SMPL FK + recompute deltas)
        5. dict_to_tensor → normalize → store as new history
        6. Update transf_rotmat/transf_transl accumulators
        """
        h_len = engine.history_shape[0]
        combined = torch.cat(
            [history_normalized, future_normalized], dim=1
        )  # [B, H+F, 276]
        combined_denorm = engine.denormalize(combined)  # [B, H+F, 276]
        raw_history = combined_denorm[:, -h_len:, :]  # [B, H, 276]

        # Convert to feature dict
        feature_dict = putil.tensor_to_dict(raw_history)

        # Attach metadata needed by get_blended_feature
        feature_dict["gender"] = self.config.gender
        feature_dict["betas"] = self._betas.unsqueeze(1).expand(
            -1, h_len, -1
        )  # [B, H, 10]
        feature_dict["transf_rotmat"] = self._transf_rotmat  # [B, 3, 3]
        feature_dict["transf_transl"] = self._transf_transl  # [B, 1, 3]
        feature_dict["pelvis_delta"] = self._pelvis_delta  # [B, 3]

        # Canonicalize + recompute features
        _, new_feature_dict = putil.get_blended_feature(
            feature_dict, use_predicted_joints=False
        )

        # Update accumulators
        self._transf_rotmat = new_feature_dict["transf_rotmat"]
        self._transf_transl = new_feature_dict["transf_transl"]

        # Convert back to tensor and normalize
        new_history = putil.dict_to_tensor(new_feature_dict)  # [B, H, 276]
        self._history = engine.normalize(new_history)

    def _get_world_features(
        self,
        engine: DartControlInference,
        putil: PrimitiveUtility,
        future_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Transform future features to world space for visualization.

        Returns: [B, F, 276] denormalized features in world coordinates.
        """
        future_denorm = engine.denormalize(future_normalized)  # [B, F, 276]
        F = future_denorm.shape[1]

        feature_dict = putil.tensor_to_dict(future_denorm)
        feature_dict["gender"] = self.config.gender
        feature_dict["betas"] = self._betas.unsqueeze(1).expand(-1, F, -1)
        feature_dict["transf_rotmat"] = self._transf_rotmat
        feature_dict["transf_transl"] = self._transf_transl
        feature_dict["pelvis_delta"] = self._pelvis_delta

        world_dict = putil.transform_feature_to_world(
            feature_dict, use_predicted_joints=True
        )
        world_tensor = putil.dict_to_tensor(world_dict)  # [B, F, 276]
        return world_tensor.cpu()

    def _get_global_joints(self, putil: PrimitiveUtility) -> torch.Tensor:
        """Get global joints from current history state.

        Returns: [B, H, 22, 3] global joint positions.
        """
        engine = self._engine
        assert engine is not None
        history_denorm = engine.denormalize(self._history)  # [B, H, 276]
        feature_dict = putil.tensor_to_dict(history_denorm)
        local_joints = feature_dict["joints"]  # [B, H, 22*3]
        B, T, _ = local_joints.shape
        local_joints = local_joints.view(B, T, 22, 3)
        transf_rotmat = self._transf_rotmat
        transf_transl = self._transf_transl
        global_joints = torch.einsum(
            "bij,btkj->btki", transf_rotmat, local_joints
        ) + transf_transl.unsqueeze(1)
        return global_joints

    def _compute_observation(
        self,
        putil: PrimitiveUtility,
        goal: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the RL policy observation vector.

        Ported from DART's env_reach_location_mld.py get_observation().

        Args:
            putil: PrimitiveUtility instance
            goal: [B, 3] goal location in world coordinates
            text_embedding: [B, 512] CLIP text embedding

        Returns:
            observation: [B, obs_dim] observation vector
        """
        device = self.config.device
        B = self.config.batch_size
        engine = self._engine
        assert engine is not None

        global_joints = self._get_global_joints(putil)  # [B, H, 22, 3]
        global_pelvis = global_joints[:, -1, 0]  # [B, 3]

        # Goal direction (XY plane)
        global_goal_dir = goal - global_pelvis  # [B, 3]
        global_goal_dir[:, 2] = 0
        goal_dist = torch.norm(global_goal_dir, dim=-1, keepdim=True)  # [B, 1]
        global_goal_dir = global_goal_dir / goal_dist.clip(min=1e-12)

        # Body forward direction
        body_orient, _ = get_new_coordinate(
            global_joints[:, -1]
        )  # [B, 3, 3], [B, 1, 3]
        forward_dir = body_orient[:, :, 1]  # [B, 3]
        forward_dir[:, 2] = 0
        moving_dir = forward_dir / torch.norm(forward_dir, dim=-1, keepdim=True).clip(
            min=1e-12
        )

        # Clip goal angle
        cos_theta = torch.einsum("bi,bi->b", global_goal_dir, moving_dir)  # [B]
        cos_theta = cos_theta.clip(
            min=np.cos(np.deg2rad(self.config.obs_goal_angle_clip)), max=1
        )
        sign = torch.sign(torch.cross(moving_dir, global_goal_dir, dim=1)[:, 2])  # [B]
        theta = torch.acos(cos_theta) * sign  # [B]
        rotation_matrix = transforms.euler_angles_to_matrix(
            torch.cat([torch.zeros(B, 2, device=device), theta.unsqueeze(1)], dim=1),
            "XYZ",
        )  # [B, 3, 3]
        global_goal_dir = torch.einsum(
            "bij,bj->bi", rotation_matrix, moving_dir
        )  # [B, 3]

        # Transform to local coordinate frame
        transf_rotmat = self._transf_rotmat
        local_goal_dir = torch.einsum(
            "bij,bj->bi", transf_rotmat.permute(0, 2, 1), global_goal_dir
        )  # [B, 3]

        # Build unnormalized motion tensor from current history
        history_denorm = engine.denormalize(self._history)  # [B, H, 276]
        motion_tensor = history_denorm  # [B, H, D]

        # Floor height relative to first-frame pelvis
        floor_height = -global_joints[:, 0, 0, 2]  # [B]

        # Concatenate: goal_dir(3), goal_dist(1), text(512), motion(H*276), scene(1)
        observation = torch.cat(
            [
                local_goal_dir,  # [B, 3]
                goal_dist.clip(max=self.config.obs_goal_dist_clip),  # [B, 1]
                text_embedding,  # [B, 512]
                motion_tensor.reshape(B, -1),  # [B, H*276]
                floor_height.unsqueeze(1),  # [B, 1]
            ],
            dim=-1,
        )
        return observation

    def _load_policy(
        self, engine: DartControlInference
    ) -> PolicyReachLocationMLP | None:
        """Load the RL policy if a checkpoint is configured."""
        if not self.config.policy_checkpoint:
            return None
        ckpt_path = Path(self.config.policy_checkpoint)
        if not ckpt_path.exists():
            print(f"[DartControl] Policy checkpoint not found: {ckpt_path}, skipping")
            return None

        h_len = engine.history_shape[0]
        noise_shape = engine.noise_shape
        policy_config = PolicyConfig(
            motion_dim=h_len * 276,
            action_dim=int(np.prod(noise_shape)),
        )
        policy = PolicyReachLocationMLP(policy_config).to(self.config.device)

        ckpt = torch.load(
            str(ckpt_path), map_location=self.config.device, weights_only=False
        )
        # The checkpoint may store the full agent state; extract policy weights
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Filter to only keys that exist in our stripped policy (no critic)
        model_keys = set(policy.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        policy.load_state_dict(filtered, strict=False)
        policy.eval()
        for p in policy.parameters():
            p.requires_grad = False

        print(f"[DartControl] RL policy loaded from {ckpt_path}")
        return policy

    def _generation_loop(
        self,
        engine: DartControlInference,
        putil: PrimitiveUtility,
        frame_queue: queue.Queue[dict[str, BonePose | None]],
        policy: PolicyReachLocationMLP | None,
        goal_receiver: Receiver[GoalFrame] | None,
        instruction_receiver: Receiver[TextFrame] | None,
    ) -> None:
        """Background thread: generates motion primitives and fills the queue."""
        instruction: str | None = None
        text_embedding: torch.Tensor | None = None

        # Current goal (None means no goal → random noise)
        current_goal: torch.Tensor | None = None

        # Set up non-blocking generators for input channels
        goal_gen = (
            goal_receiver(self, newest=True, no_block=True)
            if goal_receiver is not None
            else None
        )
        instruction_gen = (
            instruction_receiver(self, newest=True, no_block=True)
            if instruction_receiver is not None
            else None
        )

        print("[DartControl] Idle, waiting for instruction...")

        while not self.stop_event.is_set():
            try:
                # Poll instruction channel (non-blocking)
                if instruction_gen is not None:
                    instr_frame = next(instruction_gen)
                    if instr_frame is not None and isinstance(instr_frame, TextFrame):
                        new_instruction = instr_frame.get()
                        if new_instruction and new_instruction != instruction:
                            instruction = new_instruction
                            print(f"[DartControl] Instruction updated: '{instruction}'")
                            text_embedding = engine.encode_text([instruction])
                            text_embedding = text_embedding.expand(
                                self.config.batch_size, -1
                            )

                # Stay idle until an instruction is received
                if text_embedding is None:
                    self.stop_event.wait(0.1)
                    continue

                # Poll goal channel (non-blocking)
                if goal_gen is not None:
                    goal_frame = next(goal_gen)
                    if goal_frame is not None and isinstance(goal_frame, GoalFrame):
                        current_goal = torch.tensor(
                            [[goal_frame.x, goal_frame.y, goal_frame.z]],
                            device=self.config.device,
                            dtype=torch.float32,
                        ).expand(self.config.batch_size, -1)

                # Compute noise from policy if available
                noise: torch.Tensor | None = None
                if policy is not None and current_goal is not None:
                    obs = self._compute_observation(putil, current_goal, text_embedding)
                    action = policy.get_action_mean(obs)  # [B, action_dim]
                    noise = action.view(self.config.batch_size, *engine.noise_shape)

                # Generate future frames (normalized)
                future_normalized = engine.generate_step(
                    text_embedding=text_embedding,
                    history_motion=self._history,
                    guidance_scale=self.config.guidance_scale,
                    future_length=self.config.future_length,
                    noise=noise,
                )  # [B, F, 276] normalized

                # Transform to world space for output
                world_features = self._get_world_features(
                    engine, putil, future_normalized
                )  # [B, F, 276]

                # Update history with canonicalization pipeline (before emitting, uses old transf)
                self._update_history(engine, putil, self._history, future_normalized)

                # Emit world-space frames
                batch = world_features[0]  # [F, 276]
                for f in range(batch.shape[0]):
                    if self.stop_event.is_set():
                        break
                    pose = _features_to_body_pose(batch[f])
                    # Floor clamp: if lowest foot is below ground, shift entire skeleton up
                    foot_ys = [
                        bp.pos_y
                        for k in ("left_foot", "right_foot")
                        if (bp := pose.get(k)) is not None
                    ]
                    if foot_ys:
                        min_y = min(foot_ys)
                        if min_y < 0:
                            pose = {
                                name: (
                                    BonePose(
                                        pos_x=bp.pos_x,
                                        pos_y=bp.pos_y - min_y,
                                        pos_z=bp.pos_z,
                                        rot_w=bp.rot_w,
                                        rot_x=bp.rot_x,
                                        rot_y=bp.rot_y,
                                        rot_z=bp.rot_z,
                                    )
                                    if bp is not None
                                    else None
                                )
                                for name, bp in pose.items()
                            }
                    frame_queue.put(pose)

            except Exception as e:
                print(f"[DartControl] Generation error: {e}")
                import traceback

                traceback.print_exc()
                continue

    def run(self, inputs: DartControlInputs, outputs: DartControlOutputs) -> None:
        print("[DartControl] Starting DartControl component, loading models...")
        engine = self._ensure_engine()
        putil = self._ensure_primitive_util()
        print("[DartControl] Models loaded, initializing from stand pose...")

        # Initialize history from stand.pkl
        self._init_from_stand(engine, putil)

        # Load RL policy if configured
        policy = self._load_policy(engine)
        print("[DartControl] Streaming motion")

        frame_queue: queue.Queue[dict[str, BonePose | None]] = queue.Queue(maxsize=64)

        gen_thread = threading.Thread(
            target=self._generation_loop,
            args=(engine, putil, frame_queue, policy, inputs.goal, inputs.instruction),
            daemon=True,
        )
        gen_thread.start()

        frame_interval = 1.0 / self.config.fps
        while not self.stop_event.is_set():
            try:
                body_poses = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            outputs.motion.send(BodyPoseFrame(poses=body_poses))
            self.stop_event.wait(frame_interval)

        print("[DartControl] Component stopped")
