from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import torch

import src.lib.motion.dart_control.component as dart_mod
from src.lib.motion.dart_control.component import (
    BonePose,
    DartControl,
    DartControlConfig,
    DartControlInputs,
    DartControlOutputs,
)
from src.core.frames import GoalFrame, TextFrame


class _FakeReceiver:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self.blocking = True
        self.newest = False

    def __next__(self) -> object:
        if self._items:
            return self._items.pop(0)
        return None


class _FakeFuture:
    def __init__(self, value) -> None:
        self._value = value
        self.cancelled = False

    def result(self):
        return self._value

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _QueueExecutor:
    def __init__(self, *args, **kwargs) -> None:
        self.values: list[object] = []
        self.futures: list[_FakeFuture] = []

    def submit(self, fn, *args):
        if self.values:
            value = self.values.pop(0)
        else:
            value = fn(*args)
        future = _FakeFuture(value)
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = False) -> None:
        return None


def _make_component(**kwargs) -> DartControl:
    config = DartControlConfig(
        device="cpu", batch_size=1, future_length=2, fps=60.0, **kwargs
    )
    return DartControl(config)


def test_component_math_and_pose_helpers(monkeypatch) -> None:
    assert dart_mod._quat_multiply((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)) == (
        0.0,
        1.0,
        0.0,
        0.0,
    )

    rotmat = dart_mod._rot6d_to_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert rotmat.shape == (3, 3)

    assert len(dart_mod._rotmat_to_quaternion(torch.eye(3))) == 4
    assert (
        len(dart_mod._rotmat_to_quaternion(torch.diag(torch.tensor([2.0, -1.0, -1.0]))))
        == 4
    )
    assert (
        len(dart_mod._rotmat_to_quaternion(torch.diag(torch.tensor([-1.0, 2.0, -1.0]))))
        == 4
    )
    assert (
        len(dart_mod._rotmat_to_quaternion(torch.diag(torch.tensor([-1.0, -1.0, 2.0]))))
        == 4
    )

    monkeypatch.setattr(dart_mod, "_rot6d_to_matrix", lambda _rot6d: torch.eye(3))
    monkeypatch.setattr(
        dart_mod, "_rotmat_to_quaternion", lambda _m: (1.0, 0.0, 0.0, 0.0)
    )
    features = torch.zeros(276, dtype=torch.float32)
    joints = torch.arange(22 * 3, dtype=torch.float32).reshape(22, 3)
    features[dart_mod._JOINTS_OFFSET : dart_mod._JOINTS_OFFSET + 22 * 3] = (
        joints.reshape(-1)
    )
    poses = dart_mod._features_to_body_pose(features)
    assert poses["waist"] is not None
    assert poses["waist"].pos_x == -joints[0, 0].item()
    assert poses["waist"].pos_y == joints[0, 2].item()
    assert poses["waist"].pos_z == -joints[0, 1].item()


def test_component_lazy_loaders_and_prepare_frames(monkeypatch) -> None:
    created = {"engine": 0, "putil": 0}

    class _Engine:
        def __init__(self, **kwargs) -> None:
            created["engine"] += 1

    class _Primitive:
        def __init__(self, **kwargs) -> None:
            created["putil"] += 1

    monkeypatch.setattr(dart_mod, "DartControlInference", _Engine)
    monkeypatch.setattr(dart_mod, "PrimitiveUtility", _Primitive)

    component = _make_component()
    assert component._ensure_engine() is component._ensure_engine()
    assert component._ensure_primitive_util() is component._ensure_primitive_util()
    assert created == {"engine": 1, "putil": 1}

    monkeypatch.setattr(
        dart_mod,
        "_features_to_body_pose",
        lambda _frame: {
            "left_foot": BonePose(pos_x=0.0, pos_y=-1.0, pos_z=0.0, rot_w=1.0),
            "right_foot": BonePose(pos_x=0.0, pos_y=0.5, pos_z=0.0, rot_w=1.0),
            "waist": BonePose(pos_x=0.0, pos_y=1.0, pos_z=0.0, rot_w=1.0),
        },
    )
    frames = component._prepare_frames(torch.zeros((1, 2, 276), dtype=torch.float32))
    assert frames[0]["left_foot"] is not None
    assert frames[0]["left_foot"].pos_y == 0.0


def test_component_init_from_stand_and_history_updates(
    monkeypatch, tmp_path: Path
) -> None:
    component = _make_component(stand_path=str(tmp_path / "stand.pkl"))

    stand_data = {
        "transl": [[0.0, 0.0, 0.0]],
        "global_orient": [[0.0, 0.0, 0.0]],
        "body_pose": [[0.0] * 63],
    }
    with open(component.config.stand_path, "wb") as fh:
        pickle.dump(stand_data, fh)

    monkeypatch.setattr(
        dart_mod.transforms,
        "axis_angle_to_matrix",
        lambda values: (
            torch.eye(3, dtype=torch.float32).expand(*values.shape[:-1], 3, 3).clone()
        ),
    )

    engine = SimpleNamespace(
        history_shape=(2, 276),
        normalize=lambda value: value + 1,
        denormalize=lambda value: value - 1,
    )

    class _Primitive:
        def calc_calibrate_offset(self, body_param_dict):
            return torch.ones((1, 3), dtype=torch.float32)

        def canonicalize(self, primitive_dict):
            return (
                primitive_dict["transf_rotmat"],
                primitive_dict["transf_transl"],
                primitive_dict,
            )

        def calc_features(self, primitive_dict):
            return {
                "transl": torch.zeros((1, 3, 3), dtype=torch.float32),
                "poses_6d": torch.zeros((1, 3, 22 * 6), dtype=torch.float32),
                "transl_delta": torch.zeros((1, 2, 3), dtype=torch.float32),
                "global_orient_delta_6d": torch.zeros((1, 2, 6), dtype=torch.float32),
                "joints": torch.zeros((1, 3, 22 * 3), dtype=torch.float32),
                "joints_delta": torch.zeros((1, 2, 22 * 3), dtype=torch.float32),
            }

        def dict_to_tensor(self, feature_dict):
            return torch.cat(
                [
                    feature_dict["transl"],
                    feature_dict["poses_6d"],
                    feature_dict["transl_delta"],
                    feature_dict["global_orient_delta_6d"],
                    feature_dict["joints"],
                    feature_dict["joints_delta"],
                ],
                dim=-1,
            )

        def tensor_to_dict(self, tensor):
            return {
                "transl": tensor[..., :3],
                "poses_6d": tensor[..., 3:135],
                "transl_delta": tensor[..., 135:138],
                "global_orient_delta_6d": tensor[..., 138:144],
                "joints": tensor[..., 144:210],
                "joints_delta": tensor[..., 210:276],
            }

        def get_blended_feature(self, feature_dict, use_predicted_joints=True):
            return (
                {
                    "transf_rotmat": torch.eye(3).unsqueeze(0) * 2,
                    "transf_transl": torch.ones((1, 1, 3), dtype=torch.float32),
                },
                {
                    "transl": torch.zeros((1, 2, 3), dtype=torch.float32),
                    "poses_6d": torch.zeros((1, 2, 22 * 6), dtype=torch.float32),
                    "transl_delta": torch.zeros((1, 2, 3), dtype=torch.float32),
                    "global_orient_delta_6d": torch.zeros(
                        (1, 2, 6), dtype=torch.float32
                    ),
                    "joints": torch.zeros((1, 2, 22 * 3), dtype=torch.float32),
                    "joints_delta": torch.zeros((1, 2, 22 * 3), dtype=torch.float32),
                },
            )

        def transform_feature_to_world(self, feature_dict, use_predicted_joints=True):
            return feature_dict

    putil = _Primitive()

    component._init_from_stand(engine, putil)
    assert component._history.shape == (1, 2, 276)
    assert component._world_history.shape == (1, 2, 276)
    assert component._pelvis_delta.shape == (1, 3)

    world_features = torch.zeros((1, 2, 276), dtype=torch.float32)
    component._betas = torch.ones((1, 10), dtype=torch.float32)
    component._world_history = torch.zeros((1, 2, 276), dtype=torch.float32)
    component._pelvis_delta = torch.zeros((1, 3), dtype=torch.float32)
    component._update_history(engine, putil, world_features)
    assert component._history.shape == (1, 2, 276)
    assert component._transf_rotmat.shape == (1, 3, 3)
    assert component._transf_transl.shape == (1, 1, 3)

    component.config.gender = "male"
    component._betas = torch.ones((1, 10), dtype=torch.float32)
    component._transf_rotmat = torch.eye(3).unsqueeze(0)
    component._transf_transl = torch.zeros((1, 1, 3), dtype=torch.float32)
    component._pelvis_delta = torch.zeros((1, 3), dtype=torch.float32)
    world = component._get_world_features(
        engine, putil, torch.ones((1, 2, 276), dtype=torch.float32)
    )
    assert world.shape == (1, 2, 276)

    component._engine = engine
    component._history = torch.zeros((1, 2, 276), dtype=torch.float32)
    joints = component._get_global_joints(putil)
    assert joints.shape == (1, 2, 22, 3)


def test_component_observation_policy_and_generation(
    monkeypatch, tmp_path: Path
) -> None:
    component = _make_component(policy_checkpoint="")
    component._engine = SimpleNamespace(
        denormalize=lambda value: value,
        history_shape=(2, 276),
        noise_shape=(1, 4),
        generate_step=lambda **kwargs: torch.zeros((1, 2, 276), dtype=torch.float32),
    )
    component._history = torch.zeros((1, 2, 276), dtype=torch.float32)
    component._transf_rotmat = torch.eye(3).unsqueeze(0)
    component._transf_transl = torch.zeros((1, 1, 3), dtype=torch.float32)
    component._betas = torch.ones((1, 10), dtype=torch.float32)
    component._pelvis_delta = torch.zeros((1, 3), dtype=torch.float32)
    component._world_history = torch.zeros((1, 2, 276), dtype=torch.float32)

    monkeypatch.setattr(
        component,
        "_get_global_joints",
        lambda _putil: torch.tensor(
            [[[[0.0, 0.0, 0.0]] * 22, [[1.0, 0.0, 0.0]] * 22]],
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(
        dart_mod,
        "get_new_coordinate",
        lambda joints: (
            torch.eye(3).unsqueeze(0),
            torch.zeros((1, 1, 3), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        dart_mod.transforms,
        "euler_angles_to_matrix",
        lambda angles, convention: (
            torch.eye(3).unsqueeze(0).repeat(angles.shape[0], 1, 1)
        ),
    )

    text_embedding = torch.zeros((1, 512), dtype=torch.float32)
    observation_goal = component._compute_observation(
        SimpleNamespace(),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        text_embedding,
        None,
    )
    observation_heading = component._compute_observation(
        SimpleNamespace(),
        None,
        text_embedding,
        goal_heading=0.5,
    )
    assert observation_goal.shape[1] == 3 + 1 + 512 + (2 * 276) + 1
    assert observation_heading.shape == observation_goal.shape

    assert component._load_policy(component._engine) is None

    missing = _make_component(policy_checkpoint=str(tmp_path / "missing.pth"))
    assert missing._load_policy(component._engine) is None

    loaded = _make_component(policy_checkpoint=str(tmp_path / "policy.pth"))
    Path(loaded.config.policy_checkpoint).write_bytes(b"x")

    class _Policy:
        def __init__(self, config):
            self._params = [torch.nn.Parameter(torch.ones(1))]

        def to(self, device):
            return self

        def state_dict(self):
            return {"weight": torch.ones(1)}

        def load_state_dict(self, state, strict=False):
            self.state = state

        def eval(self):
            self.eval_called = True
            return self

        def parameters(self):
            return self._params

        def get_action_mean(self, obs):
            return torch.ones((1, 4), dtype=torch.float32)

    monkeypatch.setattr(dart_mod, "PolicyReachLocationMLP", _Policy)
    monkeypatch.setattr(
        dart_mod.torch,
        "load",
        lambda *args, **kwargs: {
            "model_state_dict": {"weight": torch.ones(1), "critic": torch.ones(1)}
        },
    )
    policy = loaded._load_policy(component._engine)
    assert policy is not None
    assert all(param.requires_grad is False for param in policy.parameters())

    loaded._last_goal_dist = 0.25
    loaded._last_heading_diff = 12.5
    loaded._compute_observation = lambda *args, **kwargs: torch.ones(
        (1, 10), dtype=torch.float32
    )  # type: ignore[method-assign]
    loaded._get_world_features = lambda *args, **kwargs: torch.zeros(
        (1, 2, 276), dtype=torch.float32
    )  # type: ignore[method-assign]
    updated = {"count": 0}
    loaded._update_history = lambda *args, **kwargs: updated.__setitem__(
        "count", updated["count"] + 1
    )  # type: ignore[method-assign]
    world_features, goal_dist, heading_diff = loaded._generate_primitive(
        component._engine,
        SimpleNamespace(),
        torch.zeros((1, 512), dtype=torch.float32),
        policy,
        torch.ones((1, 3), dtype=torch.float32),
        0.5,
    )
    assert world_features.shape == (1, 2, 276)
    assert goal_dist is not None and heading_diff is not None
    assert updated["count"] == 1

    world_features2, goal_dist2, heading_diff2 = loaded._generate_primitive(
        component._engine,
        SimpleNamespace(),
        torch.zeros((1, 512), dtype=torch.float32),
        None,
        None,
        None,
    )
    assert world_features2.shape == (1, 2, 276)
    assert goal_dist2 is None and heading_diff2 is None


def test_component_run_additional_branches(monkeypatch) -> None:
    executor = _QueueExecutor()
    component = _make_component(policy_checkpoint="")

    engine = SimpleNamespace(
        encode_text=lambda items: torch.ones((1, 4), dtype=torch.float32),
        noise_shape=(1, 4),
    )

    monkeypatch.setattr(component, "_ensure_engine", lambda: engine)
    monkeypatch.setattr(component, "_ensure_primitive_util", lambda: SimpleNamespace())
    monkeypatch.setattr(component, "_init_from_stand", lambda engine, putil: None)
    monkeypatch.setattr(component, "_load_policy", lambda engine: None)
    monkeypatch.setattr(
        component,
        "_prepare_frames",
        lambda world_features: [{"waist": BonePose(rot_w=1.0)}],
    )
    monkeypatch.setattr(component.stop_event, "wait", lambda timeout=None: None)
    monkeypatch.setattr(
        dart_mod, "ThreadPoolExecutor", lambda *args, **kwargs: executor
    )
    monkeypatch.setattr(
        "src.lib.motion.dart_control.component.torch.cuda.is_available", lambda: False
    )

    sent = []

    def _send(frame) -> None:
        sent.append(frame)
        component.stop_event.set()

    # Goal reached path
    executor.values = [(torch.zeros((1, 2, 276), dtype=torch.float32), 0.1, None)]
    component._generate_primitive = lambda *args, **kwargs: (
        torch.zeros((1, 2, 276), dtype=torch.float32),
        0.4,
        None,
    )  # type: ignore[method-assign]
    component.run(
        DartControlInputs(
            goal=_FakeReceiver([GoalFrame.new(x=1.0, z=2.0), None]),
            instruction=_FakeReceiver([TextFrame.new(text="walk"), None]),
        ),
        DartControlOutputs(motion=SimpleNamespace(send=_send)),
    )
    assert sent

    # Exception path
    component2 = _make_component(policy_checkpoint="")
    monkeypatch.setattr(component2, "_ensure_engine", lambda: engine)
    monkeypatch.setattr(component2, "_ensure_primitive_util", lambda: SimpleNamespace())
    monkeypatch.setattr(component2, "_init_from_stand", lambda engine, putil: None)
    monkeypatch.setattr(component2, "_load_policy", lambda engine: None)
    monkeypatch.setattr(component2, "_prepare_frames", lambda world_features: [])
    monkeypatch.setattr(component2.stop_event, "wait", lambda timeout=None: None)
    monkeypatch.setattr(
        dart_mod, "ThreadPoolExecutor", lambda *args, **kwargs: _QueueExecutor()
    )

    def _boom(*args, **kwargs):
        component2.stop_event.set()
        raise RuntimeError("boom")

    component2._generate_primitive = _boom  # type: ignore[method-assign]
    component2.run(
        DartControlInputs(
            goal=_FakeReceiver([None]), instruction=_FakeReceiver([None])
        ),
        DartControlOutputs(motion=SimpleNamespace(send=lambda frame: None)),
    )


def test_component_run_goal_and_heading_completion(monkeypatch) -> None:
    executor = _QueueExecutor()
    component = _make_component(policy_checkpoint="")

    encoded: list[str] = []
    engine = SimpleNamespace(
        encode_text=lambda items: (
            encoded.append(items[0]) or torch.ones((1, 4), dtype=torch.float32)
        ),
        noise_shape=(1, 4),
    )

    monkeypatch.setattr(component, "_ensure_engine", lambda: engine)
    monkeypatch.setattr(component, "_ensure_primitive_util", lambda: SimpleNamespace())
    monkeypatch.setattr(component, "_init_from_stand", lambda engine, putil: None)
    monkeypatch.setattr(component, "_load_policy", lambda engine: None)
    monkeypatch.setattr(
        component,
        "_prepare_frames",
        lambda world_features: [{"waist": BonePose(rot_w=1.0)}],
    )
    monkeypatch.setattr(component.stop_event, "wait", lambda timeout=None: None)
    monkeypatch.setattr(
        dart_mod, "ThreadPoolExecutor", lambda *args, **kwargs: executor
    )
    monkeypatch.setattr(
        "src.lib.motion.dart_control.component.torch.cuda.is_available", lambda: False
    )

    sent: list[object] = []

    def _send(frame) -> None:
        sent.append(frame)
        if len(sent) >= 3:
            component.stop_event.set()

    initial_world = torch.zeros((1, 2, 276), dtype=torch.float32)
    executor.values = [
        (initial_world, 0.1, 5.0),
        (initial_world, None, 2.0),
    ]
    component._generate_primitive = lambda *args, **kwargs: (  # type: ignore[method-assign]
        torch.zeros((1, 2, 276), dtype=torch.float32),
        0.6,
        30.0,
    )
    component.run(
        DartControlInputs(
            goal=_FakeReceiver([GoalFrame.new(x=1.0, z=2.0, heading=90.0), None, None]),
            instruction=_FakeReceiver([TextFrame.new(text="walk"), None, None]),
        ),
        DartControlOutputs(motion=SimpleNamespace(send=_send)),
    )

    assert len(sent) == 3
    assert "stand" in encoded


def test_component_run_heading_completion_and_stop_before_send(monkeypatch) -> None:
    executor = _QueueExecutor()
    component = _make_component(policy_checkpoint="")

    encoded: list[str] = []
    engine = SimpleNamespace(
        encode_text=lambda items: (
            encoded.append(items[0]) or torch.ones((1, 4), dtype=torch.float32)
        ),
        noise_shape=(1, 4),
    )

    monkeypatch.setattr(component, "_ensure_engine", lambda: engine)
    monkeypatch.setattr(component, "_ensure_primitive_util", lambda: SimpleNamespace())
    monkeypatch.setattr(component, "_init_from_stand", lambda engine, putil: None)
    monkeypatch.setattr(component, "_load_policy", lambda engine: None)

    def _prepare_frames(world_features):
        component.stop_event.set()
        return [{"waist": BonePose(rot_w=1.0)}]

    monkeypatch.setattr(component, "_prepare_frames", _prepare_frames)
    monkeypatch.setattr(component.stop_event, "wait", lambda timeout=None: None)
    monkeypatch.setattr(
        dart_mod, "ThreadPoolExecutor", lambda *args, **kwargs: executor
    )
    monkeypatch.setattr(
        "src.lib.motion.dart_control.component.torch.cuda.is_available", lambda: False
    )

    sends: list[object] = []
    executor.values = [(torch.zeros((1, 2, 276), dtype=torch.float32), None, 2.0)]
    component._generate_primitive = lambda *args, **kwargs: (  # type: ignore[method-assign]
        torch.zeros((1, 2, 276), dtype=torch.float32),
        None,
        45.0,
    )
    component.run(
        DartControlInputs(
            goal=_FakeReceiver(
                [GoalFrame.new(heading=45.0), GoalFrame.new(y=3.0), None]
            ),
            instruction=_FakeReceiver([None]),
        ),
        DartControlOutputs(
            motion=SimpleNamespace(send=lambda frame: sends.append(frame))
        ),
    )

    assert sends == []
    assert "stand" in encoded


def test_component_run_position_only_goal_completion(monkeypatch) -> None:
    executor = _QueueExecutor()
    component = _make_component(policy_checkpoint="")

    encoded: list[str] = []
    engine = SimpleNamespace(
        encode_text=lambda items: (
            encoded.append(items[0]) or torch.ones((1, 4), dtype=torch.float32)
        ),
        noise_shape=(1, 4),
    )

    monkeypatch.setattr(component, "_ensure_engine", lambda: engine)
    monkeypatch.setattr(component, "_ensure_primitive_util", lambda: SimpleNamespace())
    monkeypatch.setattr(component, "_init_from_stand", lambda engine, putil: None)
    monkeypatch.setattr(component, "_load_policy", lambda engine: None)
    monkeypatch.setattr(
        component,
        "_prepare_frames",
        lambda world_features: [{"waist": BonePose(rot_w=1.0)}],
    )
    monkeypatch.setattr(component.stop_event, "wait", lambda timeout=None: None)
    monkeypatch.setattr(
        dart_mod, "ThreadPoolExecutor", lambda *args, **kwargs: executor
    )
    monkeypatch.setattr(
        "src.lib.motion.dart_control.component.torch.cuda.is_available", lambda: False
    )

    sends: list[object] = []

    def _send(frame) -> None:
        sends.append(frame)
        if len(sends) >= 2:
            component.stop_event.set()

    executor.values = [(torch.zeros((1, 2, 276), dtype=torch.float32), 0.1, None)]
    component._generate_primitive = lambda *args, **kwargs: (  # type: ignore[method-assign]
        torch.zeros((1, 2, 276), dtype=torch.float32),
        0.6,
        None,
    )
    component.run(
        DartControlInputs(
            goal=_FakeReceiver([GoalFrame.new(x=1.0, z=2.0), None]),
            instruction=_FakeReceiver([TextFrame.new(text="walk"), None]),
        ),
        DartControlOutputs(motion=SimpleNamespace(send=_send)),
    )

    assert len(sends) == 2
    assert "stand" in encoded
