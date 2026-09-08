"""Build bounded Roban memdatasets from Ominisoma world-state NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
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
        if manifest.suffix.lower() == ".json":
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_rows = payload.get("sources", payload.get("selected", []))
            lines = [
                str(row["relative_path"] if isinstance(row, dict) else row)
                for row in manifest_rows
            ]
        else:
            lines = [
                raw_line.strip()
                for raw_line in manifest.read_text(encoding="utf-8").splitlines()
                if raw_line.strip() and not raw_line.strip().startswith("#")
            ]
        rows = []
        for line in lines:
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
        # A manifest is an ordered selection contract.  Preserve that order so
        # hash-ranked manifests are not silently changed back to filename order.
        candidates = rows
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
    target_total_frames: int | None = None,
    selection_seed: str | None = None,
) -> tuple[Path, list[Path], dict]:
    """Deterministically select bounded inputs and fully validate each one."""
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if max_files is not None and max_files <= 0:
        raise ValueError("--max-files must be positive")
    if max_total_frames is not None and max_total_frames <= 0:
        raise ValueError("--max-total-frames must be positive")
    if target_total_frames is not None and target_total_frames <= 0:
        raise ValueError("--target-total-frames must be positive")
    if max_total_frames is not None and target_total_frames is not None:
        raise ValueError(
            "--max-total-frames and --target-total-frames are mutually exclusive"
        )
    if max_files is None and max_total_frames is None and target_total_frames is None:
        raise ValueError(
            "refusing an unbounded conversion: pass --max-files, "
            "--max-total-frames, and/or --target-total-frames"
        )

    category_root, all_candidates = _resolve_candidates(dataset_root.resolve(), category, manifest)
    selection_hashes: dict[Path, str] = {}
    if selection_seed is not None:
        for path in all_candidates:
            category_relative = path.relative_to(category_root).as_posix()
            selection_hashes[path] = hashlib.sha256(
                f"{selection_seed}{category_relative}".encode("utf-8")
            ).hexdigest()
        all_candidates = sorted(
            all_candidates,
            key=lambda path: (
                selection_hashes[path],
                path.relative_to(category_root).as_posix(),
            ),
        )
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
                    "selection_hash": selection_hashes.get(path),
                    "sha256_before": digest_before,
                    "error_type": type(exc).__name__,
                    "reason": str(exc).replace(str(dataset_root.resolve()) + "/", ""),
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
                "selection_rank": start_index + attempted - 1,
                "selection_hash": selection_hashes.get(path),
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
        if target_total_frames is not None and total_frames >= target_total_frames:
            limit_reason = "target_total_frames"
            break

    if not selected:
        raise RuntimeError(
            f"no valid input fits the requested limits; attempted={attempted}, failures={len(failures)}"
        )
    if target_total_frames is not None and total_frames < target_total_frames:
        raise RuntimeError(
            f"valid inputs exhausted before target frames: {total_frames} < {target_total_frames}"
        )
    report = {
        "category": category,
        "available_files": len(all_candidates),
        "selection_algorithm": (
            "sha256(selection_seed + category_relative_path), ascending"
            if selection_seed is not None
            else "manifest order" if manifest is not None else "relative path, ascending"
        ),
        "selection_seed": selection_seed,
        "target_total_frames": target_total_frames,
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
    audit_output = (
        None
        if args.audit_output is None
        else resolve_output_path(args.audit_output)
    )
    if audit_output == output:
        raise ValueError("--audit-output must differ from --output")
    if output.exists() and not args.overwrite and not args.audit_only:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output}")
    if (
        audit_output is not None
        and audit_output.exists()
        and not args.overwrite
        and not args.audit_only
    ):
        raise FileExistsError(
            f"audit output already exists (pass --overwrite to replace it): {audit_output}"
        )

    _, selected, report = select_and_audit_inputs(
        dataset_root=dataset_root,
        category=args.category,
        manifest=args.manifest,
        start_index=args.start_index,
        max_files=args.max_files,
        max_total_frames=args.max_total_frames,
        target_total_frames=args.target_total_frames,
        selection_seed=args.selection_seed,
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
        for row in labels:
            source_path = Path(row["source_path"]).resolve()
            if not _is_relative_to(source_path, dataset_root):
                raise RuntimeError(
                    f"generated label source escapes dataset root: {source_path}"
                )
            row["source_path"] = source_path.relative_to(dataset_root).as_posix()
        labels_path.write_text(
            json.dumps(labels, ensure_ascii=True) + "\n", encoding="utf-8"
        )
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
                "audit_output": None if audit_output is None else str(audit_output),
                "actual_source_file_count": len(selected),
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
        selection_manifest = {
            "version": 1,
            "category": args.category,
            "path_base": "Ominisoma train root",
            "selection_algorithm": report["selection_algorithm"],
            "selection_seed": report["selection_seed"],
            "target_total_frames": report["target_total_frames"],
            "available_files": report["available_files"],
            "attempted_files": report["attempted_files"],
            "selected_files": report["selected_files"],
            "input_total_frames": report["input_total_frames"],
            "target_reached": (
                args.target_total_frames is None
                or report["input_total_frames"] >= args.target_total_frames
            ),
            "sources": [
                {
                    key: row[key]
                    for key in (
                        "selection_rank",
                        "selection_hash",
                        "relative_path",
                        "frames",
                        "sha256_before",
                    )
                }
                for row in report["sources"]
            ],
        }
        selection_manifest_path = temporary_output / "selection_manifest.json"
        selection_manifest_path.write_text(
            json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["selection_manifest_sha256"] = sha256_file(selection_manifest_path)
        (temporary_output / "selection_manifest.sha256").write_text(
            f"{report['selection_manifest_sha256']}  selection_manifest.json\n",
            encoding="utf-8",
        )

        source_manifest_path = temporary_output / "source_manifest.json"
        source_manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "category": args.category,
                    "path_base": "Ominisoma train root",
                    "selection_manifest_sha256": report["selection_manifest_sha256"],
                    "source_manifest_sha256": report["source_manifest_sha256"],
                    "sources": report["sources"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report["source_manifest_file_sha256"] = sha256_file(source_manifest_path)
        with (temporary_output / "failed_files.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for failure in report["failures"]:
                stream.write(json.dumps(failure, sort_keys=True) + "\n")
        report_path = temporary_output / "conversion_report.json"
        for _ in range(3):
            report["output_size_bytes"] = _directory_size(temporary_output)
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if audit_output is not None:
        artifact_names = (
            "selection_manifest.json",
            "selection_manifest.sha256",
            "conversion_report.json",
            "failed_files.jsonl",
            "source_manifest.json",
        )
        with atomic_output_directory(
            audit_output, overwrite=args.overwrite
        ) as temporary_audit:
            for name in artifact_names:
                shutil.copy2(output / name, temporary_audit / name)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-total-frames", type=int)
    parser.add_argument(
        "--target-total-frames",
        type=int,
        help="Select complete valid motions until this frame target is reached or exceeded",
    )
    parser.add_argument(
        "--selection-seed",
        type=str,
        help="Order candidates by SHA-256(selection seed + category-relative path)",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Optional atomically-published directory containing audit artifacts",
    )
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
