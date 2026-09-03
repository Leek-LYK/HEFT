"""Audit Ominisoma world-state motions against the current Roban MJCF."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from active_adaptation.assets.roban import ROBAN_JOINT_ORDER, ROBAN_XML
from active_adaptation.utils.roban_ominisoma import (
    COMMON_SOURCE_CURRENT_BODY_NAMES,
    REQUIRED_KEYS,
    SOURCE_BODY_NAMES,
    build_joint_reorder_indices,
)


CATEGORIES = ("acrobatic", "crawl", "jump", "kneel", "main", "squat")
COMMON_BODY_NAMES = COMMON_SOURCE_CURRENT_BODY_NAMES
def _uniform_frame_indices(frame_count: int, sample_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("motion has no frames")
    count = min(frame_count, max(1, int(sample_count)))
    return np.unique(np.linspace(0, frame_count - 1, count, dtype=np.int64))


def _metric(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(tuple(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _quat_geodesic(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    q1 = q1_wxyz / np.linalg.norm(q1_wxyz)
    q2 = q2_wxyz / np.linalg.norm(q2_wxyz)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _validate_shapes(path: Path, arrays: dict[str, np.ndarray]) -> int:
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    frame_count = int(arrays["joint_pos"].shape[0])
    expected = {
        "body_ang_vel_w": (frame_count, 28, 3),
        "body_lin_vel_w": (frame_count, 28, 3),
        "body_pos_w": (frame_count, 28, 3),
        "body_quat_w": (frame_count, 28, 4),
        "joint_pos": (frame_count, 21),
        "joint_vel": (frame_count, 21),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} shape {arrays[key].shape} != {shape}")
    fps = np.asarray(arrays["fps"])
    if fps.shape != (1,) or int(fps[0]) != 50:
        raise ValueError(f"{path}: fps must be shape (1,) with value 50, got {fps!r}")
    return frame_count


def _selected_paths(dataset_root: Path, files_per_category: int) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {}
    for category in CATEGORIES:
        paths = sorted((dataset_root / category).rglob("*.npz"))
        if len(paths) < files_per_category:
            raise RuntimeError(
                f"{dataset_root / category}: expected at least {files_per_category} NPZ files, got {len(paths)}"
            )
        selected[category] = paths[:files_per_category]
    return selected


def audit_dataset(
    dataset_root: Path,
    *,
    files_per_category: int = 2,
    frames_per_file: int = 16,
) -> dict:
    """Compare deterministic source samples with +1/-1 Roban waist FK."""
    model = mujoco.MjModel.from_xml_path(str(ROBAN_XML))
    data = mujoco.MjData(model)
    source_body_index = {name: index for index, name in enumerate(SOURCE_BODY_NAMES)}
    model_body_ids = {name: model.body(name).id for name in ("base_link", *COMMON_BODY_NAMES)}
    reorder = build_joint_reorder_indices()
    waist_index = ROBAN_JOINT_ORDER.index("waist_yaw_joint")

    values: dict[int, dict[str, dict[str, list[float]]]] = {
        sign: {
            body: defaultdict(list)
            for body in COMMON_BODY_NAMES
        }
        for sign in (1, -1)
    }
    aggregate: dict[int, dict[str, list[float]]] = {
        sign: defaultdict(list) for sign in (1, -1)
    }
    root_metrics: dict[int, dict[str, list[float]]] = {
        sign: defaultdict(list) for sign in (1, -1)
    }
    file_rows: list[dict] = []
    total_sampled_frames = 0

    for category, paths in _selected_paths(dataset_root, files_per_category).items():
        for path in paths:
            with np.load(path, allow_pickle=False) as source:
                arrays = {key: np.asarray(source[key]) for key in source.files}
            frame_count = _validate_shapes(path, arrays)
            numeric_keys = REQUIRED_KEYS[:-1]
            nonfinite_count = sum(
                int(np.size(arrays[key]) - np.count_nonzero(np.isfinite(arrays[key])))
                for key in numeric_keys
            )
            quat_norm = np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1)
            near_zero_quat_count = int(np.count_nonzero(quat_norm <= 1.0e-6))
            quat_norm_max_deviation = float(np.max(np.abs(quat_norm - 1.0)))
            if nonfinite_count:
                raise ValueError(f"{path}: found {nonfinite_count} NaN/Inf values")
            if near_zero_quat_count:
                raise ValueError(f"{path}: found {near_zero_quat_count} near-zero quaternions")
            if quat_norm_max_deviation > 1.0e-2:
                raise ValueError(
                    f"{path}: severe quaternion norm deviation {quat_norm_max_deviation:.6g} > 0.01"
                )
            all_dots = np.sum(
                arrays["body_quat_w"][1:].astype(np.float64)
                * arrays["body_quat_w"][:-1].astype(np.float64),
                axis=-1,
            )
            root_dots = all_dots[:, 0]
            sampled = _uniform_frame_indices(frame_count, frames_per_file)
            total_sampled_frames += int(sampled.size)
            file_rows.append(
                {
                    "category": category,
                    "path": path.relative_to(dataset_root).as_posix(),
                    "frames": frame_count,
                    "sampled_frames": sampled.tolist(),
                    "dtypes": {key: str(arrays[key].dtype) for key in REQUIRED_KEYS},
                    "nonfinite_count": nonfinite_count,
                    "quat_norm_max_deviation": quat_norm_max_deviation,
                    "quat_sign_flips": int(np.count_nonzero(all_dots < 0.0)),
                    "root_quat_sign_flips": int(np.count_nonzero(root_dots < 0.0)),
                }
            )

            canonical_joint_pos = arrays["joint_pos"][:, reorder].astype(np.float64)
            root_pos = arrays["body_pos_w"][:, 0].astype(np.float64)
            root_quat = arrays["body_quat_w"][:, 0].astype(np.float64)
            root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
            source_body_pos = arrays["body_pos_w"].astype(np.float64)
            source_body_quat = arrays["body_quat_w"].astype(np.float64)

            for frame in sampled:
                for sign in (1, -1):
                    data.qpos[:] = model.qpos0
                    data.qpos[:3] = root_pos[frame]
                    data.qpos[3:7] = root_quat[frame]
                    joints = canonical_joint_pos[frame].copy()
                    joints[waist_index] *= sign
                    for joint_name, joint_value in zip(ROBAN_JOINT_ORDER, joints):
                        data.qpos[model.joint(joint_name).qposadr] = joint_value
                    mujoco.mj_forward(model, data)

                    base_id = model_body_ids["base_link"]
                    base_pos_error = float(np.linalg.norm(data.xpos[base_id] - root_pos[frame]))
                    base_quat_error = _quat_geodesic(data.xquat[base_id], root_quat[frame])
                    root_metrics[sign]["position"].append(base_pos_error)
                    root_metrics[sign]["orientation_rad"].append(base_quat_error)

                    source_base_pos = source_body_pos[frame, 0]
                    simulated_base_pos = data.xpos[base_id]
                    for body_name in COMMON_BODY_NAMES:
                        source_idx = source_body_index[body_name]
                        body_id = model_body_ids[body_name]
                        abs_pos_error = float(
                            np.linalg.norm(data.xpos[body_id] - source_body_pos[frame, source_idx])
                        )
                        rel_pos_error = float(
                            np.linalg.norm(
                                (data.xpos[body_id] - simulated_base_pos)
                                - (source_body_pos[frame, source_idx] - source_base_pos)
                            )
                        )
                        orientation_error = _quat_geodesic(
                            data.xquat[body_id], source_body_quat[frame, source_idx]
                        )
                        values[sign][body_name]["abs_position_m"].append(abs_pos_error)
                        values[sign][body_name]["relative_position_m"].append(rel_pos_error)
                        values[sign][body_name]["orientation_rad"].append(orientation_error)
                        aggregate[sign]["abs_position_m"].append(abs_pos_error)
                        aggregate[sign]["relative_position_m"].append(rel_pos_error)
                        aggregate[sign]["orientation_rad"].append(orientation_error)

    by_sign = {}
    for sign in (1, -1):
        by_sign[str(sign)] = {
            "root": {name: _metric(metric_values) for name, metric_values in root_metrics[sign].items()},
            "aggregate": {name: _metric(metric_values) for name, metric_values in aggregate[sign].items()},
            "bodies": {
                body: {name: _metric(metric_values) for name, metric_values in body_values.items()}
                for body, body_values in values[sign].items()
            },
        }

    return {
        "contract": {
            "root_body": "base_link",
            "root_body_index": 0,
            "source_quaternion_order": "wxyz",
            "target_joint_names": list(ROBAN_JOINT_ORDER),
            "source_to_canonical_indices": list(reorder),
            "common_body_names": list(COMMON_BODY_NAMES),
        },
        "files_per_category": files_per_category,
        "frames_per_file": frames_per_file,
        "total_sampled_frames": total_sampled_frames,
        "files": file_rows,
        "waist_sign": by_sign,
    }


def _summary_text(report: dict) -> str:
    lines = [
        "Roban Ominisoma FK audit",
        f"files={len(report['files'])} sampled_frames={report['total_sampled_frames']}",
    ]
    for sign in ("1", "-1"):
        aggregate = report["waist_sign"][sign]["aggregate"]
        root = report["waist_sign"][sign]["root"]
        lines.extend(
            [
                f"waist_sign={sign}",
                f"  root_position_m={root['position']}",
                f"  root_orientation_rad={root['orientation_rad']}",
                f"  abs_position_m={aggregate['abs_position_m']}",
                f"  relative_position_m={aggregate['relative_position_m']}",
                f"  orientation_rad={aggregate['orientation_rad']}",
            ]
        )
    for row in report["files"]:
        lines.append(
            f"{row['path']}: frames={row['frames']} nonfinite={row['nonfinite_count']} "
            f"quat_norm_max_dev={row['quat_norm_max_deviation']:.8g} "
            f"sign_flips={row['quat_sign_flips']} root_sign_flips={row['root_quat_sign_flips']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument("--files-per-category", type=int, default=2)
    parser.add_argument("--frames-per-file", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    existing = [path for path in (args.output_json, args.output_text) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"audit output already exists (pass --overwrite to replace it): {existing}"
        )

    report = audit_dataset(
        args.dataset_root,
        files_per_category=args.files_per_category,
        frames_per_file=args.frames_per_file,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_text.write_text(_summary_text(report), encoding="utf-8")
    print(_summary_text(report), end="")


if __name__ == "__main__":
    main()
