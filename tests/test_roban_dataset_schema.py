from __future__ import annotations

import json

import numpy as np
import torch

from active_adaptation.assets.roban import ROBAN_JOINT_ORDER
from active_adaptation.utils.motion import GlobalMotionDataset, create_motion_dataset_from_path
from active_adaptation.utils.roban_ominisoma import (
    ROBAN_FOOT_NAMES,
    SOURCE_TO_CANONICAL_INDICES,
    load_roban_ominisoma_motion,
    sha256_file,
)
from roban_dataset_helpers import basic_source_arrays, write_source_npz


def test_synthetic_source_builds_minimal_memdataset_without_source_body_metadata(tmp_path):
    arrays = basic_source_arrays(frame_count=8)
    arrays["body_quat_w"][1::2] *= -1.0
    source = write_source_npz(tmp_path / "source.npz", arrays=arrays)
    digest_before = sha256_file(source)
    output = tmp_path / "memdataset"

    create_motion_dataset_from_path(
        str(source),
        mem_path=str(output),
        target_fps=50,
        storage_float_dtype=torch.float16,
        foot_names=ROBAN_FOOT_NAMES,
        motion_adapter=load_roban_ominisoma_motion,
    )

    assert sha256_file(source) == digest_before
    assert (output / "_tensordict").is_dir()
    assert (output / "meta_motion.json").is_file()
    assert (output / "id_label.json").is_file()
    meta = json.loads((output / "meta_motion.json").read_text(encoding="utf-8"))
    labels = json.loads((output / "id_label.json").read_text(encoding="utf-8"))
    assert meta["starts"] == [0]
    assert meta["ends"] == [8]
    assert meta["joint_names"] == list(ROBAN_JOINT_ORDER)
    assert meta["body_names"] == []
    assert len(labels) == 1

    dataset = GlobalMotionDataset.load_from_mem_paths([str(output)])
    assert dataset.num_motions == 1
    assert dataset.num_steps == 8
    assert tuple(dataset.data.root_pos_w.shape) == (8, 3)
    assert tuple(dataset.data.root_quat_w.shape) == (8, 4)
    assert tuple(dataset.data.joint_pos.shape) == (8, 21)
    assert dataset.joint_names == list(ROBAN_JOINT_ORDER)
    assert dataset.body_names == []
    assert torch.isfinite(dataset.data.root_pos_w).all()
    assert torch.isfinite(dataset.data.root_quat_w).all()
    assert torch.isfinite(dataset.data.joint_pos).all()
    torch.testing.assert_close(
        dataset.data.root_quat_w.norm(dim=-1), torch.ones(8), atol=1.0e-3, rtol=0.0
    )
    np.testing.assert_allclose(
        dataset.data.joint_pos.numpy(),
        arrays["joint_pos"][:, SOURCE_TO_CANONICAL_INDICES],
        atol=1.0e-2,
    )
