from __future__ import annotations

from types import SimpleNamespace

import torch

import src.lib.motion.dart_control.smpl_utils as smpl_utils


class _FakeBodyModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(
        self,
        betas=None,
        global_orient=None,
        body_pose=None,
        transl=None,
        return_vertices=False,
        **kwargs,
    ):
        if transl is not None:
            batch = transl.shape[0]
            base = transl.reshape(batch, 3)
        else:
            batch = betas.shape[0]
            base = torch.zeros((batch, 3), dtype=torch.float32)
        joints = torch.zeros((batch, 150, 3), dtype=torch.float32)
        joints[:, 0, :] = base
        joints[:, 1, :] = torch.tensor([1.0, 0.0, 0.0])
        joints[:, 2, :] = torch.tensor([0.0, 1.0, 0.0])
        vertices = torch.ones((batch, 5, 3), dtype=torch.float32)
        return SimpleNamespace(joints=joints, vertices=vertices)


def _patch_transforms(monkeypatch) -> None:
    def axis_angle_to_matrix(values: torch.Tensor) -> torch.Tensor:
        prefix = values.shape[:-1]
        eye = torch.eye(3, dtype=torch.float32)
        return eye.expand(*prefix, 3, 3).clone()

    def rotation_6d_to_matrix(values: torch.Tensor) -> torch.Tensor:
        prefix = values.shape[:-1]
        eye = torch.eye(3, dtype=torch.float32)
        return eye.expand(*prefix, 3, 3).clone()

    def matrix_to_rotation_6d(values: torch.Tensor) -> torch.Tensor:
        prefix = values.shape[:-2]
        return torch.zeros((*prefix, 6), dtype=torch.float32)

    monkeypatch.setattr(
        smpl_utils.transforms, "axis_angle_to_matrix", axis_angle_to_matrix
    )
    monkeypatch.setattr(
        smpl_utils.transforms, "rotation_6d_to_matrix", rotation_6d_to_matrix
    )
    monkeypatch.setattr(
        smpl_utils.transforms, "matrix_to_rotation_6d", matrix_to_rotation_6d
    )


def _feature_dict(batch: int = 1, frames: int = 2) -> dict[str, torch.Tensor | str]:
    return {
        "gender": "male",
        "betas": torch.ones((batch, frames, 10), dtype=torch.float32),
        "transf_rotmat": torch.eye(3, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch, 1, 1),
        "transf_transl": torch.zeros((batch, 1, 3), dtype=torch.float32),
        "pelvis_delta": torch.zeros((batch, 3), dtype=torch.float32),
        "transl": torch.zeros((batch, frames, 3), dtype=torch.float32),
        "poses_6d": torch.zeros((batch, frames, 22 * 6), dtype=torch.float32),
        "transl_delta": torch.zeros((batch, frames, 3), dtype=torch.float32),
        "global_orient_delta_6d": torch.zeros((batch, frames, 6), dtype=torch.float32),
        "joints": torch.zeros((batch, frames, 22 * 3), dtype=torch.float32),
        "joints_delta": torch.zeros((batch, frames, 22 * 3), dtype=torch.float32),
    }


def test_smpl_utils_top_level_and_basic_primitive_utility(monkeypatch) -> None:
    _patch_transforms(monkeypatch)
    monkeypatch.setattr(
        smpl_utils.smplx, "build_layer", lambda *args, **kwargs: _FakeBodyModel()
    )

    tensor_dict = {"a": torch.ones(1), "b": "x"}
    converted = smpl_utils.tensor_dict_to_device(dict(tensor_dict), "cpu")
    assert converted["a"].device.type == "cpu"

    aa_dict = {
        "global_orient": torch.zeros((1, 3), dtype=torch.float32),
        "body_pose": torch.zeros((1, 21, 3), dtype=torch.float32),
    }
    rotmat_dict = smpl_utils.convert_smpl_aa_to_rotmat(aa_dict)
    assert rotmat_dict["global_orient"].shape == (1, 3, 3)
    assert rotmat_dict["body_pose"].shape == (1, 21, 3, 3)

    primitive = {
        "gender": "male",
        "transl": torch.zeros((2, 3), dtype=torch.float32),
        "betas": torch.ones((10,), dtype=torch.float32),
        "poses_6d": torch.zeros((2, 22 * 6), dtype=torch.float32),
    }
    body_param = smpl_utils.get_smplx_param_from_6d(primitive)
    assert body_param["betas"].shape == (2, 10)
    primitive["betas"] = torch.ones((2, 10), dtype=torch.float32)
    assert smpl_utils.get_smplx_param_from_6d(primitive)["betas"].shape == (2, 10)

    joints = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    new_rotmat, new_transl = smpl_utils.get_new_coordinate(joints)
    assert new_rotmat.shape == (1, 3, 3)
    assert new_transl.shape == (1, 1, 3)

    updated = smpl_utils.update_global_transform(
        {
            "transf_rotmat": torch.eye(3).unsqueeze(0),
            "transf_transl": torch.zeros((1, 1, 3)),
        },
        torch.eye(3).unsqueeze(0),
        torch.ones((1, 1, 3)),
    )
    assert updated["transf_transl"].shape == (1, 1, 3)

    local_points = torch.ones((1, 2, 3), dtype=torch.float32)
    global_points = smpl_utils.transform_local_points_to_global(
        local_points,
        torch.eye(3).unsqueeze(0),
        torch.zeros((1, 1, 3)),
    )
    assert global_points.shape == (1, 2, 3)
    assert smpl_utils.transform_global_points_to_local(
        global_points,
        torch.eye(3).unsqueeze(0),
        torch.zeros((1, 1, 3)),
    ).shape == (1, 2, 3)

    subset = smpl_utils.get_dict_subset_by_batch(
        {
            "gender": "male",
            "transl": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        },
        0,
    )
    assert subset["gender"] == "male"
    assert subset["transl"].shape == (3,)

    utility = smpl_utils.PrimitiveUtility(device="cpu")
    assert utility.get_smpl_model("male") is utility.bm_male
    assert utility.get_smpl_model("female") is utility.bm_female

    feature_dict = _feature_dict()
    tensor = utility.dict_to_tensor(feature_dict)
    assert utility.tensor_to_dict(tensor)["transl"].shape == (1, 2, 3)

    smpl_dict = utility.feature_dict_to_smpl_dict(feature_dict)
    assert smpl_dict["global_orient"].shape == (1, 2, 3, 3)

    vertices = utility.smpl_dict_to_vertices(
        {
            "gender": "male",
            "betas": torch.ones((1, 2, 10), dtype=torch.float32),
            "global_orient": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "body_pose": torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 2, 21, 1, 1),
            "transl": torch.zeros((1, 2, 3), dtype=torch.float32),
        }
    )
    assert vertices.shape == (1, 2, 5, 3)

    joints_only = utility.smpl_dict_inference(
        {
            "gender": "male",
            "betas": torch.ones((3, 10), dtype=torch.float32),
            "global_orient": torch.eye(3).reshape(1, 3, 3).repeat(3, 1, 1),
            "body_pose": torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 21, 1, 1),
            "transl": torch.zeros((3, 3), dtype=torch.float32),
        },
        return_vertices=False,
        batch_size=2,
    )
    joints_and_vertices = utility.smpl_dict_inference(
        {
            "gender": "male",
            "betas": torch.ones((3, 10), dtype=torch.float32),
            "global_orient": torch.eye(3).reshape(1, 3, 3).repeat(3, 1, 1),
            "body_pose": torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 21, 1, 1),
            "transl": torch.zeros((3, 3), dtype=torch.float32),
        },
        return_vertices=True,
        batch_size=2,
    )
    assert joints_only.shape == (3, 22, 3)
    assert joints_and_vertices[0].shape == (3, 22, 3)
    assert joints_and_vertices[1].shape == (3, 5, 3)


def test_smpl_utils_advanced_primitive_paths(monkeypatch) -> None:
    _patch_transforms(monkeypatch)
    monkeypatch.setattr(
        smpl_utils.smplx, "build_layer", lambda *args, **kwargs: _FakeBodyModel()
    )

    utility = smpl_utils.PrimitiveUtility(device="cpu")
    body_param = {
        "gender": "male",
        "betas": torch.ones((1, 10), dtype=torch.float32),
        "transl": torch.zeros((1, 3), dtype=torch.float32),
        "body_pose": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 21, 1, 1),
        "global_orient": torch.eye(3).reshape(1, 3, 3),
    }

    rotmat_a, transl_a = utility.get_new_coordinate(
        body_param, use_predicted_joints=False
    )
    rotmat_b, transl_b = utility.get_new_coordinate(
        body_param,
        use_predicted_joints=True,
        pred_joints=torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        ),
    )
    assert rotmat_a.shape == transl_a.new_zeros((1, 3, 3)).shape
    assert rotmat_b.shape == transl_b.new_zeros((1, 3, 3)).shape
    assert utility.calc_calibrate_offset(body_param).shape == (1, 3)

    feature_dict = _feature_dict(batch=1, frames=2)
    primitive_dict = utility.feature_dict_to_smpl_dict(feature_dict)
    primitive_dict["gender"] = "male"
    primitive_dict["betas"] = feature_dict["betas"]
    primitive_dict["transf_rotmat"] = feature_dict["transf_rotmat"]
    primitive_dict["transf_transl"] = feature_dict["transf_transl"]
    primitive_dict["pelvis_delta"] = feature_dict["pelvis_delta"]
    primitive_dict["joints"] = feature_dict["joints"]

    transf_rotmat, transf_transl, canonicalized = utility.canonicalize(
        primitive_dict,
        use_predicted_joints=True,
    )
    assert transf_rotmat.shape == (1, 3, 3)
    assert transf_transl.shape == (1, 1, 3)
    assert canonicalized["transl"].shape == (1, 2, 3)

    features_pred = utility.calc_features(canonicalized, use_predicted_joints=True)
    features_smpl = utility.calc_features(canonicalized, use_predicted_joints=False)
    assert features_pred["transl"].shape == (1, 2, 3)
    assert features_smpl["poses_6d"].shape == (1, 2, 22 * 6)

    primitive_from_pred, blended_pred = utility.get_blended_feature(
        feature_dict,
        use_predicted_joints=True,
    )
    primitive_from_smpl, blended_smpl = utility.get_blended_feature(
        feature_dict,
        use_predicted_joints=False,
    )
    assert primitive_from_pred["transl"].shape == (1, 2, 3)
    assert blended_pred["joints"].shape == (1, 2, 22 * 3)
    assert primitive_from_smpl["transl"].shape == (1, 2, 3)
    assert blended_smpl["global_orient_delta_6d"].shape == (1, 2, 6)

    world_feature_dict = utility.transform_feature_to_world(feature_dict)
    assert world_feature_dict["transl"].shape == (1, 2, 3)

    primitive_world = utility.transform_primitive_to_world(
        {
            "gender": "male",
            "betas": feature_dict["betas"],
            "transf_rotmat": feature_dict["transf_rotmat"],
            "transf_transl": feature_dict["transf_transl"],
            "pelvis_delta": feature_dict["pelvis_delta"],
            "transl": torch.zeros((1, 2, 3), dtype=torch.float32),
            "global_orient": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "body_pose": torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 2, 21, 1, 1),
            "joints": torch.zeros((1, 2, 22, 3), dtype=torch.float32),
        }
    )
    assert primitive_world["transf_rotmat"].shape == (1, 3, 3)
    assert primitive_world["transf_transl"].shape == (1, 1, 3)
