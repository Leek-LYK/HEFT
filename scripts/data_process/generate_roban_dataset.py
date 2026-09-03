"""Build bounded Roban memdatasets from Ominisoma world-state NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from functools import partial
from pathlib import Path
from typing import Sequence

import torch

from active_adaptation.utils.motion import create_motion_dataset_from_path
from active_adaptation.utils.roban_ominisoma import (
    ROBAN_FOOT_NAMES,
    atomic_output_directory,
    load_roban_ominisoma_motion,
    resolve_confined_output_path,
    sha256_file,
)
if __package__:
    from .generate_dataset import check_motion, preprocess_motion
else:  # Direct execution: python scripts/data_process/generate_roban_dataset.py
    from generate_dataset import check_motion, preprocess_motion


CATEGORIES = ("acrobatic", "crawl", "jump", "kneel", "main", "squat")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA_ROOT = PROJECT_ROOT / "dataset"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_output_path(output: Path) -> Path:
    """Return an absolute output path confined to this checkout's dataset tree."""
    return resolve_confined_output_path(output, output_root=PROJECT_DATA_ROOT)


def _resolve_candidates(
    dataset_root: Path,
    category: str,
    manifest: Path | None,
) -> tuple[Path, list[Path]]:
    category_root = (dataset_root / category).resolve()
    if not category_root.is_dir():
        raise FileNotFoundError(f"category directory does not exist: {category_root}")
    if manifest is None:
        candidates = sorted(category_root.rglob("*.npz"))
    else:
        rows = []
        for raw_line in manifest.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            relative = Path(line)
            if relative.is_absolute():
                raise ValueError(f"manifest entries must be relative paths, got {line!r}")
            candidate = (
                dataset_root / relative
                if relative.parts and relative.parts[0] == category
                else category_root / relative
            ).resolve()
            if not _is_relative_to(candidate, category_root):
                raise ValueError(f"manifest entry escapes category directory: {line!r}")
            rows.append(candidate)
        candidates = sorted(rows)
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"manifest contains duplicate paths: {manifest}")
    if not candidates:
        raise RuntimeError(f"no NPZ files selected under {category_root}")
    missing = [str(path) for path in candidates if not path.is_file() or path.suffix != ".npz"]
    if missing:
        raise FileNotFoundError(f"manifest references missing/non-NPZ files: {missing}")
    return category_root, candidates


def _manifest_digest(rows: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['relative_path']}\t{row['sha256_before']}\n".encode("utf-8"))
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def select_and_audit_inputs(
    *,
    dataset_root: Path,
    category: str,
    manifest: Path | None,
    start_index: int,
    max_files: int | None,
    max_total_frames: int | None,
) -> tuple[Path, list[Path], dict]:
    """Deterministically select bounded inputs and fully validate each one."""
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if max_files is not None and max_files <= 0:
        raise ValueError("--max-files must be positive")
    if max_total_frames is not None and max_total_frames <= 0:
        raise ValueError("--max-total-frames must be positive")
    if max_files is None and max_total_frames is None:
        raise ValueError(
            "refusing an unbounded conversion: pass --max-files and/or --max-total-frames"
        )

    category_root, all_candidates = _resolve_candidates(dataset_root.resolve(), category, manifest)
    candidates = all_candidates[start_index:]
    selected: list[Path] = []
    source_rows: list[dict] = []
    failures: list[dict] = []
    total_frames = 0
    attempted = 0
    consumed = 0
    limit_reason = None

    for path in candidates:
        if max_files is not None and attempted >= max_files:
            limit_reason = "max_files"
            break
        attempted += 1
        relative_path = path.relative_to(dataset_root.resolve()).as_posix()
        digest_before = sha256_file(path)
        try:
            motion = load_roban_ominisoma_motion(path)
        except Exception as exc:
            failures.append(
                {
                    "relative_path": relative_path,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
            consumed += 1
            continue
        frame_count = int(motion["qpos"].shape[0])
        if max_total_frames is not None and total_frames + frame_count > max_total_frames:
            limit_reason = "max_total_frames"
            del motion
            break
        selected.append(path)
        consumed += 1
        total_frames += frame_count
        source_rows.append(
            {
                "relative_path": relative_path,
                "frames": frame_count,
                "sha256_before": digest_before,
                "quat_norm_max_deviation": float(
                    motion["metadata"]["source_quat_norm_max_deviation"]
                ),
                "quat_sign_flip_count": int(
                    motion["metadata"]["source_quat_sign_flip_count"]
                ),
                "root_quat_sign_flip_count": int(
                    motion["metadata"]["source_root_quat_sign_flip_count"]
                ),
            }
        )
        del motion

    if not selected:
        raise RuntimeError(
            f"no valid input fits the requested limits; attempted={attempted}, failures={len(failures)}"
        )
    report = {
        "category": category,
        "available_files": len(all_candidates),
        "start_index": start_index,
        "attempted_files": attempted,
        "consumed_files": consumed,
        "next_start_index": start_index + consumed,
        "selected_files": len(selected),
        "input_total_frames": total_frames,
        "limit_reason": limit_reason,
        "failed_source_files": len(failures),
        "failures": failures,
        "sources": source_rows,
        "source_manifest_sha256": _manifest_digest(source_rows),
    }
    return category_root, selected, report


def build_dataset(args: argparse.Namespace) -> dict:
    """Run bounded validation and atomically publish a Roban memdataset."""
    dataset_root = args.dataset_root.resolve()
    output = resolve_output_path(args.output)
    if output.exists() and not args.overwrite and not args.audit_only:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output}")

    _, selected, report = select_and_audit_inputs(
        dataset_root=dataset_root,
        category=args.category,
        manifest=args.manifest,
        start_index=args.start_index,
        max_files=args.max_files,
        max_total_frames=args.max_total_frames,
    )
    report["audit_only"] = bool(args.audit_only)
    if args.audit_only:
        return report

    preprocess = partial(preprocess_motion, always_on_ground=False)
    with atomic_output_directory(output, overwrite=args.overwrite) as temporary_output:
        create_motion_dataset_from_path(
            str(dataset_root / args.category),
            target_fps=50,
            mem_path=str(temporary_output),
            motion_processer=preprocess,
            motion_filter=check_motion,
            storage_float_dtype=torch.float16,
            storage_int_dtype=torch.int32,
            allow_body_subset=False,
            bad_interval_padding=10,
            min_segment_frames=150,
            max_segments_per_motion=2,
            foot_names=ROBAN_FOOT_NAMES,
            motion_adapter=load_roban_ominisoma_motion,
            motion_paths=selected,
        )

        meta_path = temporary_output / "meta_motion.json"
        labels_path = temporary_output / "id_label.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        successful_paths = {Path(row["source_path"]).resolve() for row in labels}
        skipped_paths = [
            path.relative_to(dataset_root).as_posix()
            for path in selected
            if path.resolve() not in successful_paths
        ]

        hash_mismatches = []
        source_by_relative = {row["relative_path"]: row for row in report["sources"]}
        for path in selected:
            relative_path = path.relative_to(dataset_root).as_posix()
            digest_after = sha256_file(path)
            source_by_relative[relative_path]["sha256_after"] = digest_after
            if digest_after != source_by_relative[relative_path]["sha256_before"]:
                hash_mismatches.append(relative_path)
        if hash_mismatches:
            raise RuntimeError(f"source NPZ hashes changed during conversion: {hash_mismatches}")

        report.update(
            {
                "audit_only": False,
                "output": str(output),
                "successful_source_files": len(successful_paths),
                "skipped_source_files": len(skipped_paths),
                "skipped_paths": skipped_paths,
                "output_motion_count": len(meta["starts"]),
                "output_total_frames": int(meta["ends"][-1]),
                "joint_names": meta["joint_names"],
                "dataset_body_names": meta.get("body_names", []),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
        )
        (temporary_output / "source_manifest.json").write_text(
            json.dumps(
                {
                    "category": args.category,
                    "source_manifest_sha256": report["source_manifest_sha256"],
                    "sources": report["sources"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = temporary_output / "conversion_report.json"
        for _ in range(3):
            report["output_size_bytes"] = _directory_size(temporary_output)
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-total-frames", type=int)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional newline-delimited, dataset-root- or category-relative NPZ paths",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_dataset(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
