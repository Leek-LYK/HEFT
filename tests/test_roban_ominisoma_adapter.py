from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.assets.roban import ROBAN_JOINT_ORDER
from active_adaptation.utils.roban_ominisoma import (
    ROBAN_FOOT_NAMES,
    SOURCE_BODY_NAMES,
    SOURCE_JOINT_NAMES,
    SOURCE_TO_CANONICAL_INDICES,
    build_joint_reorder_indices,
    load_roban_ominisoma_motion,
    local_to_world_body_positions,
    normalize_and_continuize_quaternions,
    world_to_local_body_positions,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from roban_dataset_helpers import basic_source_arrays, write_source_npz


def test_source_schema_root_and_joint_reorder(tmp_path):
    arrays = basic_source_arrays(frame_count=4)
    root_quat = np.asarray([0.5, 0.5, -0.5, 0.5], dtype=np.float32)
    arrays["body_quat_w"][:, 0] = root_quat
    source = write_source_npz(tmp_path / "motion.npz", arrays=arrays)

    motion = load_roban_ominisoma_motion(source)

    assert motion["qpos"].shape == (4, 28)
    np.testing.assert_allclose(motion["qpos"][:, :3], arrays["body_pos_w"][:, 0])
    np.testing.assert_allclose(
        motion["qpos"][:, 3:7], np.repeat(root_quat[None, :], 4, axis=0)
    )
    np.testing.assert_allclose(
        motion["qpos"][:, 7:], arrays["joint_pos"][:, SOURCE_TO_CANONICAL_INDICES]
    )
    np.testing.assert_allclose(
        motion["joint_vel"], arrays["joint_vel"][:, SOURCE_TO_CANONICAL_INDICES]
    )
    assert motion["joint_names"] == list(ROBAN_JOINT_ORDER)
    assert motion["body_names"] == list(SOURCE_BODY_NAMES)
    assert motion["dataset_body_names"] == []
    assert tuple(ROBAN_FOOT_NAMES) == ("leg_l6_link", "leg_r6_link")


def test_quaternion_order_roundtrip_and_sign_continuity():
    quaternion = np.asarray(
        [[0.5, 0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, -0.5]], dtype=np.float32
    )
    np.testing.assert_array_equal(xyzw_to_wxyz(wxyz_to_xyzw(quaternion)), quaternion)

    continuous, quality = normalize_and_continuize_quaternions(quaternion)
    np.testing.assert_allclose(continuous[0], continuous[1])
    assert quality.sign_flip_count == 1
    assert quality.root_sign_flip_count == 1
    np.testing.assert_allclose(np.linalg.norm(continuous, axis=-1), 1.0)

    slightly_scaled = quaternion[:1] * np.float32(1.0001)
    normalized, scaled_quality = normalize_and_continuize_quaternions(slightly_scaled)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=-1), 1.0, atol=1.0e-7)
    assert 0.0 < scaled_quality.max_norm_deviation < 1.0e-2


def test_quaternion_quality_rejects_zero_nonfinite_and_severe_norm(tmp_path):
    arrays = basic_source_arrays(frame_count=3)
    arrays["body_quat_w"][1, 0] = 0.0
    path = write_source_npz(tmp_path / "zero.npz", arrays=arrays)
    with pytest.raises(ValueError, match="near-zero"):
        load_roban_ominisoma_motion(path)

    arrays = basic_source_arrays(frame_count=3)
    arrays["body_quat_w"][1, 0, 0] = np.nan
    path = write_source_npz(tmp_path / "nan.npz", arrays=arrays)
    with pytest.raises(ValueError, match="NaN/Inf"):
        load_roban_ominisoma_motion(path)

    arrays = basic_source_arrays(frame_count=3)
    arrays["body_quat_w"][1, 0, 0] = 1.1
    path = write_source_npz(tmp_path / "norm.npz", arrays=arrays)
    with pytest.raises(ValueError, match="severe quaternion norm deviation"):
        load_roban_ominisoma_motion(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda arrays: arrays.pop("body_pos_w"), "missing required keys"),
        (
            lambda arrays: arrays.__setitem__("body_pos_w", arrays["body_pos_w"][:, :-1]),
            "body_pos_w shape",
        ),
        (
            lambda arrays: arrays.__setitem__("joint_pos", arrays["joint_pos"][:, :-1]),
            "joint_pos shape",
        ),
        (
            lambda arrays: arrays.__setitem__("joint_vel", arrays["joint_vel"][:-1]),
            "joint_vel shape",
        ),
        (
            lambda arrays: arrays.__setitem__("fps", np.asarray([60], dtype=np.int64)),
            "expected fps=50",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "joint_pos", arrays["joint_pos"].astype(np.complex64)
            ),
            "cannot be safely interpreted as real values",
        ),
    ],
)
def test_schema_errors_identify_the_bad_field(tmp_path, mutation, message):
    arrays = basic_source_arrays(frame_count=3)
    mutation(arrays)
    path = write_source_npz(tmp_path / f"bad-{message.split()[0]}.npz", arrays=arrays)
    with pytest.raises(ValueError, match=message):
        load_roban_ominisoma_motion(path)


def test_joint_mapping_is_name_derived_and_rejects_contract_errors():
    assert SOURCE_TO_CANONICAL_INDICES == (
        0,
        3,
        7,
        11,
        15,
        17,
        19,
        4,
        8,
        12,
        16,
        18,
        20,
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
    )
    source_missing = tuple(name for name in SOURCE_JOINT_NAMES if name != "leg_r6")
    with pytest.raises(ValueError, match="21 names"):
        build_joint_reorder_indices(source_missing)
    source_duplicate = SOURCE_JOINT_NAMES[:-1] + (SOURCE_JOINT_NAMES[-2],)
    with pytest.raises(ValueError, match="source_duplicates"):
        build_joint_reorder_indices(source_duplicate)
    source_unknown = SOURCE_JOINT_NAMES[:-1] + ("unknown",)
    with pytest.raises(ValueError, match="missing=.*leg_r6.*unknown=.*unknown"):
        build_joint_reorder_indices(source_unknown)


def test_local_body_positions_use_inverse_root_rotation_and_roundtrip():
    root_pos = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    root_xyzw = sRot.from_euler("z", 90.0, degrees=True).as_quat().astype(np.float32)
    root_wxyz = xyzw_to_wxyz(root_xyzw)[None, :]
    body_pos = np.asarray([[[1.0, 3.0, 3.0], [0.0, 2.0, 4.0]]], dtype=np.float32)

    local = world_to_local_body_positions(body_pos, root_pos, root_wxyz)

    np.testing.assert_allclose(local[0, 0], [1.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(
        local_to_world_body_positions(local, root_pos, root_wxyz), body_pos, atol=1.0e-6
    )


def test_legacy_body_names_are_not_renamed(tmp_path):
    motion = load_roban_ominisoma_motion(write_source_npz(tmp_path / "motion.npz"))
    assert "zarm_l5_link" in motion["body_names"]
    assert "zarm_r5_link" in motion["body_names"]
    assert "camera_base" in motion["body_names"]
    assert "l_handball" not in motion["body_names"]
    assert "r_handball" not in motion["body_names"]
    assert "head_camera_base" not in motion["body_names"]
