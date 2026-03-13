"""PoseRenderer3D conduit — renders BodyPoseFrame as a 3D SMPL body mesh."""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import numpy as np
import torch
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import BodyPoseFrame, BonePose, VideoFrame, VideoDataFormat


# SMPL joint index → OpenVR body part name (same mapping as DartControl)
_SMPL_TO_OPENVR: dict[int, str] = {
    0: "waist",
    4: "left_knee",
    5: "right_knee",
    9: "chest",
    10: "left_foot",
    11: "right_foot",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_hand",
    21: "right_hand",
}
_OPENVR_TO_SMPL: dict[str, int] = {v: k for k, v in _SMPL_TO_OPENVR.items()}

# SMPLX kinematic tree parents
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
_N_JOINTS = 22

# Quaternion for converting Y-up back to Z-up (+90° around X, inverse of Z-up→Y-up)
_SQRT2_2 = math.sqrt(2.0) / 2.0
_Q_YUP_TO_ZUP = (_SQRT2_2, _SQRT2_2, 0.0, 0.0)


def _quat_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


class PoseRenderer3DConfig(BaseModel):
    width: int = 640
    height: int = 480
    gender: str = "male"
    device: str = "cpu"
    camera_distance: float = 3.5
    camera_height: float = 0.8


class PoseRenderer3DInputs(NamedTuple):
    pose: Receiver[BodyPoseFrame]


class PoseRenderer3DOutputs(NamedTuple):
    video: Sender[VideoFrame]


class PoseRenderer3D(Component[PoseRenderer3DInputs, PoseRenderer3DOutputs]):
    """Renders BodyPoseFrame as a 3D SMPL body mesh using pyrender."""

    def __init__(self, config: PoseRenderer3DConfig) -> None:
        super().__init__()
        self.config = config
        self._initialized = False

    def _ensure_resources(self) -> None:
        if self._initialized:
            return

        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        import pyrender
        import trimesh
        import smplx
        from pathlib import Path

        self._pyrender = pyrender
        self._trimesh = trimesh

        # Load SMPL body model
        body_model_dir = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "dart_control"
            / "smplx_models"
        )
        self._body_model = (
            smplx.build_layer(
                str(body_model_dir),
                model_type="smplx",
                gender=self.config.gender,
                ext="npz",
                num_pca_comps=12,
            )
            .to(self.config.device)
            .eval()
        )
        self._faces = self._body_model.faces.astype(np.int32)

        # Set up pyrender renderer and scene
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=self.config.width,
            viewport_height=self.config.height,
            point_size=0.5,
        )
        self._scene = pyrender.Scene(
            bg_color=[0.1, 0.1, 0.1, 1.0],
            ambient_light=(0.4, 0.4, 0.4),
        )

        # Add lights (same setup as DART renderer)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        for pos in [[0, -1, 1], [0, 1, 1], [1, 1, 2]]:
            light_pose = np.eye(4)
            light_pose[:3, 3] = pos
            self._scene.add(light, pose=light_pose)

        self._initialized = True
        print("[PoseRenderer3D] Initialized SMPL model and pyrender scene")

    def _body_pose_to_smpl_params(
        self, poses: dict[str, BonePose | None]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert BodyPoseFrame joints to SMPL parameters.

        Inverts the FK chain: given world-space quaternions (Y-up), converts
        back to Z-up, then computes local rotations for each joint.

        Returns:
            global_orient: [3, 3] rotation matrix
            body_pose: [21, 3, 3] local rotation matrices
            transl: [3] root translation in Z-up
        """
        # Get world-space Z-up rotation matrices for available joints
        world_rots_zup: dict[int, np.ndarray] = {}
        for part_name, bone in poses.items():
            if bone is None:
                continue
            smpl_idx = _OPENVR_TO_SMPL.get(part_name)
            if smpl_idx is None:
                continue
            q_yup = (bone.rot_w, bone.rot_x, bone.rot_y, bone.rot_z)
            q_zup = _quat_multiply(_Q_YUP_TO_ZUP, q_yup)
            world_rots_zup[smpl_idx] = _quat_to_rotmat(*q_zup)

        # Build all 22 world rotations in topological order
        # For missing joints, inherit parent's world rotation (= identity local rotation)
        world_rots: list[np.ndarray] = [np.eye(3, dtype=np.float32)] * _N_JOINTS
        for j in range(_N_JOINTS):
            if j in world_rots_zup:
                world_rots[j] = world_rots_zup[j]
            else:
                parent = _SMPL_PARENTS[j]
                if parent >= 0:
                    world_rots[j] = world_rots[parent].copy()

        # Invert FK: local_rot[j] = parent_world.T @ world[j]
        global_orient = world_rots[0]
        body_pose = np.zeros((21, 3, 3), dtype=np.float32)
        for j in range(1, _N_JOINTS):
            parent = _SMPL_PARENTS[j]
            body_pose[j - 1] = world_rots[parent].T @ world_rots[j]

        # Translation: waist position converted Y-up → Z-up
        # Y-up (x, y, z) → Z-up: x_z = x_y, y_z = -z_y, z_z = y_y
        waist = poses.get("waist")
        if waist is not None:
            transl = np.array(
                [waist.pos_x, -waist.pos_z, waist.pos_y], dtype=np.float32
            )
        else:
            transl = np.zeros(3, dtype=np.float32)

        return global_orient, body_pose, transl

    @staticmethod
    def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Compute a camera-to-world 4x4 matrix (pyrender convention)."""
        f = target - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        mat = np.eye(4)
        mat[:3, 0] = s
        mat[:3, 1] = u
        mat[:3, 2] = -f
        mat[:3, 3] = eye
        return mat

    def _render_mesh(self, vertices: np.ndarray) -> np.ndarray:
        """Render SMPL mesh to an RGB image."""
        pyrender = self._pyrender

        mesh = self._trimesh.Trimesh(
            vertices=vertices, faces=self._faces, process=False
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(1.0, 1.0, 0.9, 1.0),
        )

        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = self._scene.add(render_mesh, "mesh")

        # RGB axis gizmo at origin: X=red, Y=green, Z=blue
        axis_nodes = []
        axis_len = 0.3
        axis_radius = 0.008
        axes = [
            ([1, 0, 0], (1.0, 0.0, 0.0, 1.0)),  # X red
            ([0, 1, 0], (0.0, 1.0, 0.0, 1.0)),  # Y green
            ([0, 0, 1], (0.0, 0.0, 1.0, 1.0)),  # Z blue
        ]
        for direction, color in axes:
            cyl = self._trimesh.creation.cylinder(
                radius=axis_radius,
                height=axis_len,
                sections=8,
            )
            # Cylinder is along Z by default — rotate to target axis
            d = np.array(direction, dtype=np.float64)
            z = np.array([0, 0, 1], dtype=np.float64)
            if np.allclose(d, z):
                R = np.eye(4)
            elif np.allclose(d, -z):
                R = self._trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            else:
                axis = np.cross(z, d)
                angle = np.arccos(np.dot(z, d))
                R = self._trimesh.transformations.rotation_matrix(angle, axis)
            # Shift so cylinder starts at origin
            T = np.eye(4)
            T[:3, 3] = d * axis_len / 2
            cyl.apply_transform(T @ R)
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode="OPAQUE",
                baseColorFactor=color,
            )
            axis_nodes.append(
                self._scene.add(pyrender.Mesh.from_trimesh(cyl, material=mat), "axis")
            )

        # Camera looks at mesh center from the front (+Y in SMPL's Z-up space)
        center = mesh.bounds.mean(axis=0)
        eye = center + np.array([0.0, -self.config.camera_distance, 0.0])
        cam_pose = self._look_at(eye, center, up=np.array([0.0, 0.0, 1.0]))

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, znear=0.1, zfar=100.0)
        cam_node = self._scene.add(camera, pose=cam_pose)

        rgb, _ = self._renderer.render(self._scene, flags=pyrender.RenderFlags.RGBA)
        image = rgb[:, :, :3].astype(np.uint8)

        self._scene.remove_node(mesh_node)
        self._scene.remove_node(cam_node)
        for n in axis_nodes:
            self._scene.remove_node(n)

        return image

    def run(self, inputs: PoseRenderer3DInputs, outputs: PoseRenderer3DOutputs) -> None:
        self._ensure_resources()
        device = self.config.device

        for frame in inputs.pose(self):
            if frame is None:
                break

            poses = frame.get()
            global_orient, body_pose, transl = self._body_pose_to_smpl_params(poses)

            with torch.no_grad():
                output = self._body_model(
                    betas=torch.zeros(1, 10, device=device),
                    global_orient=torch.tensor(
                        global_orient, dtype=torch.float32, device=device
                    ).unsqueeze(0),
                    body_pose=torch.tensor(
                        body_pose, dtype=torch.float32, device=device
                    ).unsqueeze(0),
                    transl=torch.tensor(
                        transl, dtype=torch.float32, device=device
                    ).unsqueeze(0),
                )

            vertices = output.vertices[0].cpu().numpy()
            image = self._render_mesh(vertices)
            outputs.video.send(VideoFrame.new(data=image, format=VideoDataFormat.RGB))
