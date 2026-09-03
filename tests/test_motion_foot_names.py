from __future__ import annotations

import json

import numpy as np
import pytest

from active_adaptation.utils.motion import (
    DEFAULT_FOOT_NAMES,
    FOOT_NAMES,
    create_motion_dataset_from_path,
    resolve_foot_indices,
)
from active_adaptation.utils.roban_ominisoma import ROBAN_FOOT_NAMES, SOURCE_BODY_NAMES


def test_default_foot_names_preserve_g1_l7_behavior():
    body_names = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
    assert DEFAULT_FOOT_NAMES == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert FOOT_NAMES == list(DEFAULT_FOOT_NAMES)
    assert resolve_foot_indices(body_names) == [1, 2]


def test_roban_foot_names_are_configured_without_robot_special_case():
    assert resolve_foot_indices(SOURCE_BODY_NAMES, ROBAN_FOOT_NAMES) == [26, 27]


@pytest.mark.parametrize(
    ("foot_names", "message"),
    [
        (("left",), "exactly two"),
        (("left", "left"), "unique"),
        (("left", "missing"), "missing from the motion body schema"),
    ],
)
def test_invalid_foot_configuration_fails_clearly(foot_names, message):
    with pytest.raises(ValueError, match=message):
        resolve_foot_indices(("left", "right"), foot_names)


def test_standard_npz_caller_keeps_default_foot_and_body_schema(tmp_path):
    source = tmp_path / "standard.npz"
    frame_count = 3
    root_rot_xyzw = np.zeros((frame_count, 4), dtype=np.float32)
    root_rot_xyzw[:, 3] = 1.0
    np.savez(
        source,
        root_pos=np.zeros((frame_count, 3), dtype=np.float32),
        root_rot=root_rot_xyzw,
        dof_pos=np.zeros((frame_count, 1), dtype=np.float32),
        local_body_pos=np.zeros((frame_count, 2, 3), dtype=np.float32),
        joint_names=np.asarray(["joint"]),
        body_names=np.asarray(DEFAULT_FOOT_NAMES),
        fps=np.asarray([50], dtype=np.int64),
    )
    output = tmp_path / "standard_memdataset"

    create_motion_dataset_from_path(str(source), mem_path=str(output))

    meta = json.loads((output / "meta_motion.json").read_text(encoding="utf-8"))
    assert meta["joint_names"] == ["joint"]
    assert meta["body_names"] == list(DEFAULT_FOOT_NAMES)
