from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

from src.core.frames import BodyPoseFrame, BonePose, VideoDataFormat


class _FakeRecv:
    def __init__(self, items):
        self._items = items

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def test_pose_renderer_3d_paths(monkeypatch) -> None:
    import src.core.conduit.pose_renderer_3d as pose3d_mod

    q = pose3d_mod._quat_multiply((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
    assert q == (0.0, 1.0, 0.0, 0.0)
    rot = pose3d_mod._quat_to_rotmat(1.0, 0.0, 0.0, 0.0)
    assert np.allclose(rot, np.eye(3, dtype=np.float32))

    monkeypatch.setattr(sys, "platform", "linux", raising=False)

    class _FakeBodyModel:
        def __init__(self):
            self.faces = np.array([[0, 1, 2]], dtype=np.int32)
            self.calls = []

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(vertices=torch.zeros((1, 1, 3), dtype=torch.float32))

    fake_body_model = _FakeBodyModel()

    class _FakeRenderer:
        def __init__(self, viewport_width, viewport_height, point_size):
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height
            self.point_size = point_size

        def render(self, scene, flags=None):
            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[..., :3] = 123
            rgba[..., 3] = 255
            return rgba, None

    class _FakeScene:
        def __init__(self, bg_color=None, ambient_light=None):
            self.bg_color = bg_color
            self.ambient_light = ambient_light
            self.nodes = []

        def add(self, obj, name=None, pose=None):
            node = SimpleNamespace(obj=obj, name=name, pose=pose)
            self.nodes.append(node)
            return node

        def remove_node(self, node):
            self.nodes.remove(node)

    class _FakeTrimeshMesh:
        def __init__(self, vertices=None, faces=None, process=False):
            self.vertices = np.asarray(vertices)
            self.faces = np.asarray(faces)
            self.process = process
            self.bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
            self.transforms = []

        def apply_transform(self, mat):
            self.transforms.append(mat)

    fake_pyrender = SimpleNamespace(
        OffscreenRenderer=_FakeRenderer,
        Scene=_FakeScene,
        PointLight=lambda color, intensity: SimpleNamespace(color=color, intensity=intensity),
        MetallicRoughnessMaterial=lambda **kwargs: SimpleNamespace(**kwargs),
        Mesh=SimpleNamespace(from_trimesh=lambda mesh, material=None: SimpleNamespace(mesh=mesh, material=material)),
        PerspectiveCamera=lambda **kwargs: SimpleNamespace(**kwargs),
        RenderFlags=SimpleNamespace(RGBA="rgba"),
    )
    fake_trimesh = SimpleNamespace(
        Trimesh=_FakeTrimeshMesh,
        creation=SimpleNamespace(cylinder=lambda radius, height, sections: _FakeTrimeshMesh(vertices=np.zeros((2, 3)), faces=np.zeros((1, 3), dtype=np.int32))),
        transformations=SimpleNamespace(rotation_matrix=lambda angle, axis: np.eye(4, dtype=np.float32)),
    )
    fake_smplx = SimpleNamespace(build_layer=lambda *args, **kwargs: fake_body_model)
    monkeypatch.setitem(sys.modules, "pyrender", fake_pyrender)
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)
    monkeypatch.setitem(sys.modules, "smplx", fake_smplx)

    renderer = pose3d_mod.PoseRenderer3D(
        pose3d_mod.PoseRenderer3DConfig(width=4, height=4, camera_distance=2.0, device="cpu")
    )
    renderer._ensure_resources()
    assert renderer._initialized is True
    assert renderer._body_model is fake_body_model
    renderer._ensure_resources()

    poses = {
        "waist": BonePose(pos_x=1.0, pos_y=2.0, pos_z=3.0),
        "chest": BonePose(rot_w=1.0, rot_x=0.0, rot_y=0.0, rot_z=0.0),
        "unknown": BonePose(),
        "none": None,
    }
    global_orient, body_pose, transl = renderer._body_pose_to_smpl_params(poses)
    assert global_orient.shape == (3, 3)
    assert body_pose.shape == (21, 3, 3)
    assert np.allclose(transl, np.array([1.0, -3.0, 2.0], dtype=np.float32))

    _, _, transl_zero = renderer._body_pose_to_smpl_params({"head": BonePose()})
    assert np.allclose(transl_zero, np.zeros(3, dtype=np.float32))

    eye = np.array([0.0, -2.0, 0.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    look = renderer._look_at(eye, target, up)
    assert look.shape == (4, 4)

    image = renderer._render_mesh(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert image.shape == (4, 4, 3)
    assert image.dtype == np.uint8

    sent = []
    renderer.run(
        pose3d_mod.PoseRenderer3DInputs(pose=_FakeRecv([BodyPoseFrame(poses=poses), None])),
        pose3d_mod.PoseRenderer3DOutputs(video=SimpleNamespace(send=lambda value: sent.append(value))),
    )
    assert len(sent) == 1
    assert sent[0].get(VideoDataFormat.RGB).shape == (4, 4, 3)
