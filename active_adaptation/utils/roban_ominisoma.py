"""Adapter contract for Leju Roban Ominisoma/IsaacLab world-state NPZ files."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.assets.roban import ROBAN_JOINT_ORDER


SOURCE_JOINT_NAMES = (
    "waist_yaw",
    "zarm_l1",
    "zarm_r1",
    "leg_l1",
    "leg_r1",
    "zarm_l2",
    "zarm_r2",
    "leg_l2",
    "leg_r2",
    "zarm_l3",
    "zarm_r3",
    "leg_l3",
    "leg_r3",
    "zarm_l4",
    "zarm_r4",
    "leg_l4",
    "leg_r4",
    "leg_l5",
    "leg_r5",
    "leg_l6",
    "leg_r6",
)
SOURCE_BODY_NAMES = (
    "base_link",
    "waist_yaw_link",
    "zarm_l1_link",
    "zarm_r1_link",
    "zhead_1_link",
    "leg_l1_link",
    "leg_r1_link",
    "zarm_l2_link",
    "zarm_r2_link",
    "zhead_2_link",
    "head_radar",
    "leg_l2_link",
    "leg_r2_link",
    "zarm_l3_link",
    "zarm_r3_link",
    "camera_base",
    "leg_l3_link",
    "leg_r3_link",
    "zarm_l4_link",
    "zarm_r4_link",
    "leg_l4_link",
    "leg_r4_link",
    "zarm_l5_link",
    "zarm_r5_link",
    "leg_l5_link",
    "leg_r5_link",
    "leg_l6_link",
    "leg_r6_link",
)
COMMON_SOURCE_CURRENT_BODY_NAMES = (
    "waist_yaw_link",
    "leg_l1_link",
    "leg_l2_link",
    "leg_l3_link",
    "leg_l4_link",
    "leg_l5_link",
    "leg_l6_link",
    "leg_r1_link",
    "leg_r2_link",
    "leg_r3_link",
    "leg_r4_link",
    "leg_r5_link",
    "leg_r6_link",
    "zarm_l1_link",
    "zarm_l2_link",
    "zarm_l3_link",
    "zarm_l4_link",
    "zarm_r1_link",
    "zarm_r2_link",
    "zarm_r3_link",
    "zarm_r4_link",
    "zhead_1_link",
    "zhead_2_link",
)
ROBAN_FOOT_NAMES = ("leg_l6_link", "leg_r6_link")
REQUIRED_KEYS = (
    "body_ang_vel_w",
    "body_lin_vel_w",
    "body_pos_w",
    "body_quat_w",
    "joint_pos",
    "joint_vel",
    "fps",
)


@dataclass(frozen=True)
class RobanOminisomaContract:
    """Fixed schema facts supplied for the legacy Ominisoma export."""

    source_joint_names: tuple[str, ...] = SOURCE_JOINT_NAMES
    source_body_names: tuple[str, ...] = SOURCE_BODY_NAMES
    root_body_name: str = "base_link"
    root_body_index: int = 0
    source_quat_order: Literal["wxyz"] = "wxyz"
    foot_names: tuple[str, str] = ROBAN_FOOT_NAMES
    fps: int = 50


@dataclass(frozen=True)
class QuaternionQuality:
    """Quaternion normalization and temporal-sign audit values."""

    max_norm_deviation: float
    sign_flip_count: int
    root_sign_flip_count: int


DEFAULT_CONTRACT = RobanOminisomaContract()


def build_joint_reorder_indices(
    source_joint_names: Sequence[str] = SOURCE_JOINT_NAMES,
    target_joint_names: Sequence[str] = ROBAN_JOINT_ORDER,
) -> tuple[int, ...]:
    """Build the source-to-canonical permutation from joint names."""
    source_names = tuple(source_joint_names)
    target_names = tuple(target_joint_names)
    if len(source_names) != 21:
        raise ValueError(f"source joint contract must contain 21 names, got {len(source_names)}")
    if len(target_names) != 21:
        raise ValueError(f"target joint contract must contain 21 names, got {len(target_names)}")
    duplicate_source = sorted({name for name in source_names if source_names.count(name) > 1})
    duplicate_target = sorted({name for name in target_names if target_names.count(name) > 1})
    if duplicate_source or duplicate_target:
        raise ValueError(
            f"joint names must be unique: source_duplicates={duplicate_source}, "
            f"target_duplicates={duplicate_target}"
        )

    source_name_to_index = {name: index for index, name in enumerate(source_names)}
    target_source_names = tuple(name.removesuffix("_joint") for name in target_names)
    missing = [name for name in target_source_names if name not in source_name_to_index]
    unknown = sorted(set(source_names) - set(target_source_names))
    if missing or unknown:
        raise ValueError(f"joint contract mismatch: missing={missing}, unknown={unknown}")
    return tuple(source_name_to_index[name] for name in target_source_names)


SOURCE_TO_CANONICAL_INDICES = build_joint_reorder_indices()


def wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion array at the adapter boundary."""
    quat = np.asarray(quaternion)
    if quat.shape[-1:] != (4,):
        raise ValueError(f"quaternion last dimension must be 4, got {quat.shape}")
    return quat[..., (1, 2, 3, 0)]


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a standard HEFT quaternion array back to internal wxyz."""
    quat = np.asarray(quaternion)
    if quat.shape[-1:] != (4,):
        raise ValueError(f"quaternion last dimension must be 4, got {quat.shape}")
    return quat[..., (3, 0, 1, 2)]


def normalize_and_continuize_quaternions(
    quaternion_wxyz: np.ndarray,
    *,
    path: Path | None = None,
    severe_norm_deviation: float = 1.0e-2,
) -> tuple[np.ndarray, QuaternionQuality]:
    """Normalize wxyz quaternions and remove equivalent q/-q time flips."""
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    context = str(path) if path is not None else "quaternion array"
    if quat.ndim < 2 or quat.shape[-1] != 4:
        raise ValueError(f"{context}: expected quaternion shape [T,...,4], got {quat.shape}")
    if not np.isfinite(quat).all():
        raise ValueError(f"{context}: quaternion array contains NaN/Inf")
    norms = np.linalg.norm(quat, axis=-1)
    near_zero = int(np.count_nonzero(norms <= 1.0e-6))
    if near_zero:
        raise ValueError(f"{context}: quaternion array contains {near_zero} near-zero values")
    max_deviation = float(np.max(np.abs(norms - 1.0)))
    if max_deviation > severe_norm_deviation:
        raise ValueError(
            f"{context}: severe quaternion norm deviation {max_deviation:.8g} "
            f"> {severe_norm_deviation:.8g}"
        )

    normalized = quat / norms[..., None]
    adjacent_dot = np.sum(normalized[1:] * normalized[:-1], axis=-1)
    flip_mask = adjacent_dot < 0.0
    sign_steps = np.where(flip_mask, -1.0, 1.0)
    factors = np.concatenate(
        [np.ones((1,) + sign_steps.shape[1:], dtype=np.float64), sign_steps], axis=0
    )
    normalized *= np.cumprod(factors, axis=0)[..., None]
    quality = QuaternionQuality(
        max_norm_deviation=max_deviation,
        sign_flip_count=int(np.count_nonzero(flip_mask)),
        root_sign_flip_count=(
            int(np.count_nonzero(flip_mask[..., 0])) if flip_mask.ndim > 1 else int(np.count_nonzero(flip_mask))
        ),
    )
    return normalized.astype(np.float32), quality


def world_to_local_body_positions(
    body_pos_w: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Express world body positions in the rotated root frame."""
    body_pos = np.asarray(body_pos_w, dtype=np.float64)
    root_pos = np.asarray(root_pos_w, dtype=np.float64)
    root_quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape [T,B,3], got {body_pos.shape}")
    if root_pos.shape != (body_pos.shape[0], 3):
        raise ValueError(f"root_pos_w shape {root_pos.shape} is incompatible with {body_pos.shape}")
    if root_quat.shape != (body_pos.shape[0], 4):
        raise ValueError(f"root_quat_wxyz shape {root_quat.shape} is incompatible with {body_pos.shape}")
    rotation = sRot.from_quat(wxyz_to_xyzw(root_quat)).as_matrix()
    delta_w = body_pos - root_pos[:, None, :]
    return np.einsum("tji,tbj->tbi", rotation, delta_w).astype(np.float32)


def local_to_world_body_positions(
    local_body_pos: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Reconstruct world body positions from root-frame positions."""
    local_pos = np.asarray(local_body_pos, dtype=np.float64)
    root_pos = np.asarray(root_pos_w, dtype=np.float64)
    root_quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    if local_pos.ndim != 3 or local_pos.shape[-1] != 3:
        raise ValueError(f"local_body_pos must have shape [T,B,3], got {local_pos.shape}")
    if root_pos.shape != (local_pos.shape[0], 3):
        raise ValueError(f"root_pos_w shape {root_pos.shape} is incompatible with {local_pos.shape}")
    if root_quat.shape != (local_pos.shape[0], 4):
        raise ValueError(f"root_quat_wxyz shape {root_quat.shape} is incompatible with {local_pos.shape}")
    rotation = sRot.from_quat(wxyz_to_xyzw(root_quat)).as_matrix()
    return (
        np.einsum("tij,tbj->tbi", rotation, local_pos) + root_pos[:, None, :]
    ).astype(np.float32)


def _safe_float32(array: np.ndarray, *, path: Path, key: str) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{path}: {key} dtype {array.dtype} cannot be safely interpreted as real values")
    if not np.can_cast(array.dtype, np.float32, casting="safe"):
        raise ValueError(f"{path}: {key} dtype {array.dtype} cannot be safely converted to float32")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: {key} contains NaN/Inf")
    return result


def _load_required_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as source:
            missing = [key for key in REQUIRED_KEYS if key not in source.files]
            if missing:
                raise ValueError(f"{path}: missing required keys {missing}")
            return {key: np.asarray(source[key]) for key in REQUIRED_KEYS}
    except (OSError, ValueError) as exc:
        if str(path) in str(exc):
            raise
        raise ValueError(f"{path}: failed to read NPZ: {exc}") from exc


def load_roban_ominisoma_motion(
    path: str | Path,
    *,
    contract: RobanOminisomaContract = DEFAULT_CONTRACT,
) -> dict:
    """Validate one source NPZ and return HEFT preprocessing arrays."""
    source_path = Path(path)
    if len(contract.source_body_names) != 28 or len(set(contract.source_body_names)) != 28:
        raise ValueError("source body contract must contain 28 unique names")
    if not 0 <= contract.root_body_index < len(contract.source_body_names):
        raise ValueError(f"root body index is out of range: {contract.root_body_index}")
    if contract.source_body_names[contract.root_body_index] != contract.root_body_name:
        raise ValueError(
            "root body contract mismatch: "
            f"index {contract.root_body_index} is "
            f"{contract.source_body_names[contract.root_body_index]!r}, "
            f"not {contract.root_body_name!r}"
        )
    if contract.source_quat_order != "wxyz":
        raise ValueError(
            f"unsupported source quaternion order {contract.source_quat_order!r}; expected 'wxyz'"
        )
    if len(set(contract.foot_names)) != 2 or any(
        name not in contract.source_body_names for name in contract.foot_names
    ):
        raise ValueError(
            f"source foot contract must contain two unique source bodies, got {contract.foot_names}"
        )
    arrays = _load_required_arrays(source_path)
    joint_pos_raw = arrays["joint_pos"]
    if joint_pos_raw.ndim != 2:
        raise ValueError(f"{source_path}: joint_pos must have shape [T,21], got {joint_pos_raw.shape}")
    frame_count = int(joint_pos_raw.shape[0])
    expected_shapes = {
        "body_ang_vel_w": (frame_count, len(contract.source_body_names), 3),
        "body_lin_vel_w": (frame_count, len(contract.source_body_names), 3),
        "body_pos_w": (frame_count, len(contract.source_body_names), 3),
        "body_quat_w": (frame_count, len(contract.source_body_names), 4),
        "joint_pos": (frame_count, len(contract.source_joint_names)),
        "joint_vel": (frame_count, len(contract.source_joint_names)),
    }
    if frame_count <= 0:
        raise ValueError(f"{source_path}: motion must contain at least one frame")
    for key, expected_shape in expected_shapes.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(f"{source_path}: {key} shape {arrays[key].shape} != {expected_shape}")

    fps = np.asarray(arrays["fps"])
    if fps.shape != (1,):
        raise ValueError(f"{source_path}: fps shape {fps.shape} != (1,)")
    if not np.issubdtype(fps.dtype, np.integer):
        raise ValueError(f"{source_path}: fps dtype must be integer, got {fps.dtype}")
    if int(fps[0]) != contract.fps:
        raise ValueError(f"{source_path}: expected fps={contract.fps}, got {int(fps[0])}")

    float_arrays = {
        key: _safe_float32(arrays[key], path=source_path, key=key)
        for key in REQUIRED_KEYS
        if key != "fps"
    }
    body_quat_wxyz, quat_quality = normalize_and_continuize_quaternions(
        float_arrays["body_quat_w"], path=source_path
    )
    root_pos_w = float_arrays["body_pos_w"][:, contract.root_body_index]
    root_quat_wxyz = body_quat_wxyz[:, contract.root_body_index]
    reorder = build_joint_reorder_indices(contract.source_joint_names, ROBAN_JOINT_ORDER)
    joint_pos = float_arrays["joint_pos"][:, reorder]
    joint_vel = float_arrays["joint_vel"][:, reorder]
    qpos = np.concatenate([root_pos_w, root_quat_wxyz, joint_pos], axis=-1).astype(np.float32)
    if qpos.shape != (frame_count, 28):
        raise RuntimeError(f"{source_path}: internal qpos shape {qpos.shape} != {(frame_count, 28)}")

    return {
        "fps": contract.fps,
        "qpos": qpos,
        # Source world positions are used only by preprocessing/filter callbacks.
        "xpos": float_arrays["body_pos_w"],
        "joint_names": list(ROBAN_JOINT_ORDER),
        "body_names": list(contract.source_body_names),
        # No legacy body-state tensor is persisted. Runtime selects current asset
        # bodies and reconstructs them from root + canonical joints through FK.
        "dataset_body_names": [],
        "joint_vel": joint_vel,
        "metadata": {
            "source_quat_norm_max_deviation": quat_quality.max_norm_deviation,
            "source_quat_sign_flip_count": quat_quality.sign_flip_count,
            "source_root_quat_sign_flip_count": quat_quality.root_sign_flip_count,
        },
    }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without modifying the source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_confined_output_path(
    output: str | Path,
    *,
    output_root: str | Path,
) -> Path:
    """Return an absolute lexical path whose resolved target stays under a root."""
    root = Path(output_root).resolve()
    lexical_output = Path(os.path.abspath(os.fspath(output)))
    resolved_output = lexical_output.resolve()
    try:
        resolved_output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output must be a child of the project dataset directory {root}: "
            f"{lexical_output} resolves to {resolved_output}"
        ) from exc
    if resolved_output == root:
        raise ValueError(
            f"output must be a child of the project dataset directory {root}: "
            f"{lexical_output} resolves to {resolved_output}"
        )
    return lexical_output


@contextmanager
def atomic_output_directory(output: str | Path, *, overwrite: bool = False) -> Iterator[Path]:
    """Yield a sibling temp directory and publish it only after success."""
    output_path = Path(output)
    if not output_path.name or output_path == Path(output_path.anchor):
        raise ValueError(f"unsafe output directory: {output_path}")
    output_exists = output_path.exists() or output_path.is_symlink()
    if output_exists and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        yield temp_path
        output_exists = output_path.exists() or output_path.is_symlink()
        if output_exists and not overwrite:
            raise FileExistsError(
                f"output appeared during conversion and was not replaced: {output_path}"
            )
        if output_exists:
            backup_path = output_path.with_name(
                f".{output_path.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(output_path, backup_path)
            try:
                os.replace(temp_path, output_path)
            except BaseException:
                os.replace(backup_path, output_path)
                raise
            if backup_path.is_dir() and not backup_path.is_symlink():
                shutil.rmtree(backup_path)
            else:
                backup_path.unlink()
        else:
            os.replace(temp_path, output_path)
    except BaseException:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise
