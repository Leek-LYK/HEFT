from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from active_adaptation.assets.roban import ROBAN_JOINT_ORDER, ROBAN_XML
from active_adaptation.utils.fk_helper import MotionFKHelper
from active_adaptation.utils.motion import GlobalMotionDataset
from active_adaptation.utils.roban_ominisoma import (
    COMMON_SOURCE_CURRENT_BODY_NAMES,
    SOURCE_BODY_NAMES,
    atomic_output_directory,
    load_roban_ominisoma_motion,
    resolve_confined_output_path,
    sha256_file,
)
from roban_dataset_helpers import write_fk_source_npz


def test_fk_generated_source_recovers_canonical_qpos_and_current_asset_fk(tmp_path):
    source, canonical = write_fk_source_npz(tmp_path / "fk.npz", frame_count=12)
    prepared = load_roban_ominisoma_motion(source)
    np.testing.assert_allclose(prepared["qpos"][:, 7:], canonical, atol=1.0e-7)

    model = mujoco.MjModel.from_xml_path(str(ROBAN_XML))
    data = mujoco.MjData(model)
    source_body_index = {name: index for index, name in enumerate(SOURCE_BODY_NAMES)}
    with np.load(source, allow_pickle=False) as arrays:
        body_pos_w = arrays["body_pos_w"]
        body_quat_w = arrays["body_quat_w"]
        for frame in (0, 5, 11):
            data.qpos[:] = model.qpos0
            data.qpos[:7] = prepared["qpos"][frame, :7]
            for name, value in zip(ROBAN_JOINT_ORDER, prepared["qpos"][frame, 7:]):
                data.qpos[model.joint(name).qposadr] = value
            mujoco.mj_forward(model, data)
            for body_name in COMMON_SOURCE_CURRENT_BODY_NAMES:
                source_index = source_body_index[body_name]
                body_id = model.body(body_name).id
                np.testing.assert_allclose(
                    data.xpos[body_id], body_pos_w[frame, source_index], atol=2.0e-6
                )
                dot = abs(float(np.dot(data.xquat[body_id], body_quat_w[frame, source_index])))
                assert dot == pytest.approx(1.0, abs=2.0e-6)


def test_cli_build_is_atomic_loadable_and_fk_expandable(tmp_path):
    dataset_root = tmp_path / "dataset"
    source, canonical = write_fk_source_npz(
        dataset_root / "main" / "walk_smoke.npz", frame_count=160
    )
    digest_before = sha256_file(source)
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "dataset" / "_tmp" / f"pytest-main-one-{uuid.uuid4().hex}"
    command = [
        str(project_root / ".venv" / "bin" / "python"),
        "scripts/data_process/generate_roban_dataset.py",
        "--dataset-root",
        str(dataset_root),
        "--category",
        "main",
        "--output",
        str(output),
        "--max-files",
        "1",
        "--max-total-frames",
        "10000",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout[completed.stdout.index("{") :])

        assert report["successful_source_files"] == 1
        assert report["skipped_source_files"] == 0
        assert report["failed_source_files"] == 0
        assert report["output_motion_count"] == 1
        assert report["output_total_frames"] == 160
        assert report["consumed_files"] == 1
        assert report["next_start_index"] == 1
        assert report["failures"] == []
        assert report["output_size_bytes"] > 0
        assert sha256_file(source) == digest_before
        assert (output / "conversion_report.json").is_file()
        assert (output / "source_manifest.json").is_file()
        persisted_report = json.loads(
            (output / "conversion_report.json").read_text(encoding="utf-8")
        )
        assert persisted_report["source_manifest_sha256"] == report["source_manifest_sha256"]
        assert persisted_report["sources"][0]["sha256_before"] == digest_before
        assert persisted_report["sources"][0]["sha256_after"] == digest_before

        dataset = GlobalMotionDataset.load_from_mem_paths([str(output)])
        np.testing.assert_allclose(dataset.data.joint_pos.numpy(), canonical, atol=1.0e-3)
        minimal = dataset.get_slice(
            torch.tensor([0]), torch.tensor([0]), steps=8, target_device=torch.device("cpu")
        )
        helper = MotionFKHelper.from_mjcf_path(
            xml_path=ROBAN_XML,
            dataset_joint_names=dataset.joint_names,
            output_body_names=("base_link", "leg_l6_link", "leg_r6_link"),
            base_body_name="base_link",
        )
        expanded = helper.expand_minimal_motion(minimal, fps=50.0)
        for field_name in expanded.__annotations__:
            assert torch.isfinite(getattr(expanded, field_name)).all(), field_name

        rerun = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert rerun.returncode != 0
        assert "--overwrite" in rerun.stderr
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_interrupted_atomic_writer_never_publishes_partial_output(tmp_path):
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_output_directory(output) as temporary:
            (temporary / "partial").write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted")
    assert not output.exists()
    assert list(tmp_path.glob(".dataset.tmp-*")) == []


def test_atomic_writer_refuses_existing_output_by_default(tmp_path):
    output = tmp_path / "dataset"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        with atomic_output_directory(output):
            pass
    assert marker.read_text(encoding="utf-8") == "user data"


def test_cli_output_path_rejects_parent_escape_and_external_symlink(
    tmp_path,
):
    data_root = tmp_path / "project" / "dataset"
    data_root.mkdir(parents=True)

    valid_output = data_root / "_smoke" / "roban" / "main_one"
    assert resolve_confined_output_path(valid_output, output_root=data_root) == valid_output

    with pytest.raises(ValueError, match="project dataset directory"):
        resolve_confined_output_path(data_root / ".." / "escaped", output_root=data_root)

    external = tmp_path / "external"
    external.mkdir()
    external_link = data_root / "external_link"
    external_link.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="project dataset directory"):
        resolve_confined_output_path(external_link / "escaped", output_root=data_root)


def test_atomic_overwrite_replaces_only_exact_target(tmp_path):
    parent = tmp_path / "outputs"
    output = parent / "target"
    sibling = parent / "sibling"
    output.mkdir(parents=True)
    sibling.mkdir()
    (output / "old").write_text("old", encoding="utf-8")
    (sibling / "keep").write_text("keep", encoding="utf-8")

    with atomic_output_directory(output, overwrite=True) as temporary:
        (temporary / "new").write_text("new", encoding="utf-8")

    assert not (output / "old").exists()
    assert (output / "new").read_text(encoding="utf-8") == "new"
    assert (sibling / "keep").read_text(encoding="utf-8") == "keep"
