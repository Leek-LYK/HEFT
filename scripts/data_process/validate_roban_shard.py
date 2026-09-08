"""Validate a generated Roban memdataset and a deterministic MJCF FK sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import time
from pathlib import Path
from typing import Any

import torch

from active_adaptation.assets.roban import ROBAN_JOINT_ORDER, ROBAN_XML
from active_adaptation.utils.fk_helper import MotionFKHelper
from active_adaptation.utils.motion import GlobalMotionDataset
from active_adaptation.utils.roban_ominisoma import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_shard(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    mem_root = Path(os.environ.get("MEMPATH", "dataset")).resolve()
    raw_mem_path = Path(args.mem_path)
    shard_path = (
        raw_mem_path.resolve()
        if raw_mem_path.is_absolute()
        else (mem_root / raw_mem_path).resolve()
    )
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"validation output already exists: {output}")
    if not shard_path.is_dir():
        raise FileNotFoundError(f"shard does not exist: {shard_path}")

    selection_path = shard_path / "selection_manifest.json"
    source_manifest_path = shard_path / "source_manifest.json"
    conversion_report_path = shard_path / "conversion_report.json"
    labels_path = shard_path / "id_label.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    conversion = json.loads(conversion_report_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    storage_meta = json.loads(
        (shard_path / "_tensordict" / "meta.json").read_text(encoding="utf-8")
    )

    selection_sha256 = sha256_file(selection_path)
    if selection_sha256 != conversion["selection_manifest_sha256"]:
        raise AssertionError("selection manifest SHA-256 mismatch")
    if selection_sha256 != source_manifest["selection_manifest_sha256"]:
        raise AssertionError("source manifest references a different selection manifest")

    dataset = GlobalMotionDataset.load_from_mem_paths(
        [str(raw_mem_path)], ds_device=torch.device("cpu")
    )
    if dataset.ds_device != torch.device("cpu") or dataset.data.device != torch.device("cpu"):
        raise AssertionError("formal loader did not keep the dataset on CPU")
    data = dataset.data
    expected_shapes = {
        "root_pos_w": [dataset.num_steps, 3],
        "root_quat_w": [dataset.num_steps, 4],
        "joint_pos": [dataset.num_steps, 21],
    }
    actual_shapes = {
        "root_pos_w": list(data.root_pos_w.shape),
        "root_quat_w": list(data.root_quat_w.shape),
        "joint_pos": list(data.joint_pos.shape),
    }
    actual_dtypes = {
        "root_pos_w": str(data.root_pos_w.dtype),
        "root_quat_w": str(data.root_quat_w.dtype),
        "joint_pos": str(data.joint_pos.dtype),
        "starts": str(dataset.starts.dtype),
        "ends": str(dataset.ends.dtype),
        "lengths": str(dataset.lengths.dtype),
    }
    if actual_shapes != expected_shapes:
        raise AssertionError(f"memdataset shape mismatch: {actual_shapes}")
    if list(dataset.joint_names) != list(ROBAN_JOINT_ORDER):
        raise AssertionError("joint names differ from ROBAN_JOINT_ORDER")

    starts = dataset.starts
    ends = dataset.ends
    if starts.numel() == 0 or starts[0].item() != 0:
        raise AssertionError("starts must begin at zero")
    if not bool((ends > starts).all().item()):
        raise AssertionError("every motion interval must be non-empty")
    if starts.numel() > 1 and not torch.equal(starts[1:], ends[:-1]):
        raise AssertionError("motion intervals are not contiguous")
    if int(ends[-1].item()) != dataset.num_steps:
        raise AssertionError("final motion end does not equal frame count")
    if len(labels) != dataset.num_motions:
        raise AssertionError("motion label count mismatch")

    selected_by_path = {
        row["relative_path"]: row for row in selection["sources"]
    }
    label_sources: set[str] = set()
    for motion_id, (label, start, end) in enumerate(
        zip(labels, starts.tolist(), ends.tolist())
    ):
        relative_path = Path(str(label["source_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AssertionError(f"motion {motion_id} has a non-relative source label")
        relative = relative_path.as_posix()
        if relative not in selected_by_path:
            raise AssertionError(f"motion {motion_id} source is absent from selection manifest")
        if int(label["segment_end"]) - int(label["segment_start"]) != end - start:
            raise AssertionError(f"motion {motion_id} label length mismatch")
        if not 0 <= int(label["segment_start"]) < int(label["segment_end"]):
            raise AssertionError(f"motion {motion_id} label interval is invalid")
        if int(label["segment_end"]) > int(selected_by_path[relative]["frames"]):
            raise AssertionError(f"motion {motion_id} label exceeds its source")
        label_sources.add(relative)
    if len(label_sources) != int(conversion["successful_source_files"]):
        raise AssertionError("successful source count does not match motion labels")

    tensor_finite = {}
    for name in ("root_pos_w", "root_quat_w", "joint_pos"):
        finite = bool(torch.isfinite(getattr(data, name)).all().item())
        tensor_finite[name] = finite
        if not finite:
            raise FloatingPointError(f"{name} contains NaN/Inf")
    quat_norms = torch.linalg.vector_norm(data.root_quat_w.double(), dim=-1)
    quaternion = {
        "min_norm": float(quat_norms.min().item()),
        "max_norm": float(quat_norms.max().item()),
        "mean_norm": float(quat_norms.mean().item()),
        "max_abs_deviation_from_one": float((quat_norms - 1.0).abs().max().item()),
    }
    if quaternion["max_abs_deviation_from_one"] > 1.0e-2:
        raise AssertionError(f"root quaternion norms are unreasonable: {quaternion}")

    sample_count = min(int(args.motion_sample_count), dataset.num_motions)
    if sample_count < min(100, dataset.num_motions):
        raise ValueError("validation must sample at least 100 motions when available")
    rng = random.Random(int(args.seed))
    sampled_motion_ids = sorted(rng.sample(range(dataset.num_motions), sample_count))
    motion_ids = []
    local_frames = []
    sampled_by_motion = {}
    for motion_id in sampled_motion_ids:
        length = int(dataset.lengths[motion_id].item())
        frame_count = min(int(args.frames_per_motion), length)
        if frame_count <= 0:
            raise AssertionError(f"motion {motion_id} has no sampleable frames")
        frames = sorted(rng.sample(range(length), frame_count))
        sampled_by_motion[str(motion_id)] = frames
        motion_ids.extend([motion_id] * len(frames))
        local_frames.extend(frames)
    minimal = dataset.get_slice(
        torch.tensor(motion_ids, dtype=torch.long),
        torch.tensor(local_frames, dtype=torch.long),
        steps=1,
        target_device=torch.device("cpu"),
    )
    helper = MotionFKHelper.from_mjcf_path(
        xml_path=ROBAN_XML,
        dataset_joint_names=dataset.joint_names,
        base_body_name="base_link",
        device="cpu",
    )
    expanded = helper.expand_minimal_motion(minimal, fps=50.0)
    fk_fields = {}
    for name in expanded.__annotations__:
        tensor = getattr(expanded, name)
        finite = bool(torch.isfinite(tensor).all().item())
        fk_fields[name] = {"shape": list(tensor.shape), "finite": finite}
        if not finite:
            raise FloatingPointError(f"FK output {name} contains NaN/Inf")

    source_hash_mismatches = []
    for row in source_manifest["sources"]:
        relative_path = Path(str(row["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AssertionError("source manifest contains a non-relative path")
        source_path = (source_root / relative_path).resolve()
        try:
            source_path.relative_to(source_root)
        except ValueError as exc:
            raise AssertionError("source manifest path escapes source root") from exc
        current_hash = sha256_file(source_path)
        if not (
            current_hash == row["sha256_before"] == row.get("sha256_after")
        ):
            source_hash_mismatches.append(relative_path.as_posix())
    if source_hash_mismatches:
        raise AssertionError(
            f"source NPZ hash mismatch: {source_hash_mismatches[:10]}"
        )

    report = {
        "status": "PASS",
        "seed": int(args.seed),
        "mem_path": str(args.mem_path),
        "resolved_shard_path": str(shard_path),
        "loader": "GlobalMotionDataset.load_from_mem_paths",
        "dataset_device": str(dataset.ds_device),
        "shapes": actual_shapes,
        "dtypes": actual_dtypes,
        "joint_names_match_roban_order": True,
        "joint_names": list(dataset.joint_names),
        "body_names": list(dataset.body_names),
        "intervals": {
            "motion_count": dataset.num_motions,
            "frame_count": dataset.num_steps,
            "start_at_zero": True,
            "contiguous": True,
            "legal": True,
        },
        "motion_labels": {
            "count": len(labels),
            "unique_source_count": len(label_sources),
            "all_relative": True,
            "match_selection_manifest": True,
        },
        "tensor_finite": tensor_finite,
        "root_quaternion_norm": quaternion,
        "fk": {
            "mjcf": str(ROBAN_XML),
            "sampled_motion_count": sample_count,
            "frames_per_motion": int(args.frames_per_motion),
            "sampled_frame_count": len(motion_ids),
            "sampled_motion_frames": sampled_by_motion,
            "fields": fk_fields,
            "all_finite": True,
        },
        "source_integrity": {
            "checked_file_count": len(source_manifest["sources"]),
            "hash_mismatches": source_hash_mismatches,
            "all_unchanged": True,
        },
        "manifests": {
            "selection_manifest_sha256": selection_sha256,
            "selection_manifest_hash_matches_report": True,
            "paths_relative_to_source_root": True,
        },
        "storage": {
            "memmap_dtypes": {
                name: storage_meta[name]["dtype"]
                for name in ("root_pos_w", "root_quat_w", "joint_pos")
            },
            "directory_size_bytes": sum(
                item.stat().st_size for item in shard_path.rglob("*") if item.is_file()
            ),
            "files": {
                item.relative_to(shard_path).as_posix(): item.stat().st_size
                for item in sorted(shard_path.rglob("*"))
                if item.is_file() and item != output
            },
            "estimated_all_source_files": int(conversion["available_files"]),
            "estimated_all_source_size_bytes": round(
                int(conversion["output_size_bytes"])
                * int(conversion["available_files"])
                / int(conversion["selected_files"])
            ),
            "estimate_method": (
                "converted directory bytes scaled by available/selected source count"
            ),
        },
        "elapsed_s": time.perf_counter() - started,
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        / 1024.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem-path", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--motion-sample-count", type=int, default=100)
    parser.add_argument("--frames-per-motion", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = validate_shard(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
