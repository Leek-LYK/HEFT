from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from active_adaptation.utils.fk_helper import MotionFKHelper, MotionMinimalData, _basename
from active_adaptation.utils.math import quat_to_rot6d
from active_adaptation.utils.symmetry import SymmetryTransform, cartesian_space_symmetry


NameSwapRules = Sequence[tuple[str, str]]
DEFAULT_NAME_SWAP_RULES: tuple[tuple[str, str], ...] = (("left_", "right_"),)


def load_spec_from_xml(xml_path: Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(xml_path))


def swap_left_right(
    name: str,
    name_swap_rules: NameSwapRules = DEFAULT_NAME_SWAP_RULES,
) -> str:
    """Swap a name using ordered, bidirectional prefix pairs."""
    matches: list[tuple[str, str]] = []
    for left, right in name_swap_rules:
        if not left or not right or left == right:
            raise ValueError(f"Invalid symmetry name swap rule: {(left, right)!r}")
        if name.startswith(left):
            matches.append((left, right))
        elif name.startswith(right):
            matches.append((right, left))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous symmetry name swap for {name!r}: {matches!r}")
    if matches:
        source, target = matches[0]
        return target + name[len(source) :]
    return name


def dominant_axis_sign(axis: Sequence[float]) -> int:
    dominant_idx = max(range(3), key=lambda i: abs(axis[i]))
    return 1 if dominant_idx == 1 else -1


def base_joint_sign(joint_name: str, axis: Sequence[float]) -> int:
    lower = joint_name.lower()
    if "roll" in lower or "yaw" in lower or "arm_yaw" in lower:
        return -1
    if "pitch" in lower or "knee" in lower or "elbow" in lower:
        return 1
    return dominant_axis_sign(axis)


def parse_xml_symmetry_info(
    xml_path: Path,
) -> tuple[
    dict[str, tuple[float, float, float]], tuple[str, ...], tuple[str, ...]
]:
    root = ET.parse(xml_path).getroot()
    joint_axes: dict[str, tuple[float, float, float]] = {}
    joint_order: list[str] = []
    body_names: list[str] = []
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if body_name is not None:
            body_names.append(body_name)
        for joint in body.findall("joint"):
            joint_name = joint.get("name")
            if joint_name is None or joint.get("type", "hinge") == "free":
                continue
            axis = tuple(float(v) for v in joint.get("axis", "0 0 0").split())
            joint_axes[joint_name] = axis
            joint_order.append(joint_name)
    return joint_axes, tuple(joint_order), tuple(body_names)


def build_auto_joint_symmetry_map(
    joint_axes: dict[str, tuple[float, float, float]],
    name_swap_rules: NameSwapRules = DEFAULT_NAME_SWAP_RULES,
) -> dict[str, tuple[int, str]]:
    """Build joint swaps and signs from names and mirrored joint axes."""
    all_joint_names = set(joint_axes)
    mapping: dict[str, tuple[int, str]] = {}
    for joint_name, axis in joint_axes.items():
        mirrored_name = swap_left_right(joint_name, name_swap_rules)
        if mirrored_name not in all_joint_names:
            mirrored_name = joint_name
        # Joint axes are axial vectors. Reflection across the sagittal plane
        # therefore transforms (x, y, z) to (-x, y, -z).
        reflected_axis = (-axis[0], axis[1], -axis[2])
        mirrored_axis = joint_axes[mirrored_name]
        axis_alignment = sum(a * b for a, b in zip(reflected_axis, mirrored_axis))
        if abs(axis_alignment) < 1.0e-8:
            raise ValueError(
                f"Cannot determine symmetry sign for joint {joint_name!r} "
                f"and mirror {mirrored_name!r}: axes={axis!r}, {mirrored_axis!r}"
            )
        sign = 1 if axis_alignment > 0.0 else -1
        mapping[joint_name] = (sign, mirrored_name)
    return mapping


def build_auto_spatial_symmetry_map(
    body_names: Sequence[str],
    name_swap_rules: NameSwapRules = DEFAULT_NAME_SWAP_RULES,
) -> dict[str, str]:
    """Build a body-name symmetry map with unmatched bodies kept centered."""
    all_body_names = set(body_names)
    mapping: dict[str, str] = {}
    for body_name in body_names:
        mirrored_name = swap_left_right(body_name, name_swap_rules)
        if mirrored_name not in all_body_names:
            mirrored_name = body_name
        mapping[body_name] = mirrored_name
    return mapping


def _validate_name_swap_involution(
    names: Sequence[str], name_swap_rules: NameSwapRules
) -> None:
    for name in names:
        swapped = swap_left_right(name, name_swap_rules)
        if swap_left_right(swapped, name_swap_rules) != name:
            raise ValueError(f"Symmetry name swap is not an involution for {name!r}")


def _select_validation_bodies(
    body_names: Sequence[str], validation_body_patterns: Sequence[str] | None
) -> tuple[str, ...]:
    if validation_body_patterns is None:
        return tuple(body_names)
    selected: list[str] = []
    for pattern in validation_body_patterns:
        matched = [name for name in body_names if re.fullmatch(pattern, name)]
        if not matched:
            raise ValueError(f"Symmetry validation body pattern matched nothing: {pattern!r}")
        for name in matched:
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def _build_fk_helper(
    *,
    model,
    dataset_joint_names: Sequence[str],
    output_body_names: Sequence[str],
) -> MotionFKHelper:
    body_name_to_id = {_basename(model.body(i).name): i for i in range(model.nbody)}
    joint_id_to_name = {i: _basename(model.joint(i).name) for i in range(model.njnt)}
    return MotionFKHelper._build(
        model=model,
        body_name_to_id=body_name_to_id,
        joint_id_to_name=joint_id_to_name,
        dataset_joint_names=dataset_joint_names,
        output_body_names=list(output_body_names),
        base_body_name=_basename(model.body(1).name),
        device=torch.device("cpu"),
    )


def _validate_spatial_structure(model, spatial_map: dict[str, str]) -> tuple[float, float]:
    mirror = np.diag([1.0, -1.0, 1.0])
    body_name_to_id = {_basename(model.body(i).name): i for i in range(model.nbody)}
    body_pos = np.asarray(model.body_pos).reshape(model.nbody, 3)
    body_quat = np.asarray(model.body_quat).reshape(model.nbody, 4)

    def quat_wxyz_to_xyzw(q):
        return np.concatenate([q[..., 1:], q[..., 0:1]], axis=-1)

    pos_err = 0.0
    quat_err = 0.0
    for body_name, mirrored_name in spatial_map.items():
        if mirrored_name not in body_name_to_id or body_name not in body_name_to_id:
            continue
        i = body_name_to_id[body_name]
        j = body_name_to_id[mirrored_name]
        pos_err = max(pos_err, float(np.max(np.abs(body_pos[j] - (mirror @ body_pos[i])))))
        rot_i = R.from_quat(quat_wxyz_to_xyzw(body_quat[i])).as_matrix()
        rot_j = R.from_quat(quat_wxyz_to_xyzw(body_quat[j])).as_matrix()
        quat_err = max(quat_err, float(np.max(np.abs(rot_j - mirror @ rot_i @ mirror))))
    return pos_err, quat_err


def _validate_fk_pose_symmetry(
    *,
    model,
    helper: MotionFKHelper,
    joint_names: Sequence[str],
    body_names: Sequence[str],
    joint_map: dict[str, tuple[int, str]],
    spatial_map: dict[str, str],
) -> tuple[float, float]:
    class Cfg:
        pass

    class Asset:
        pass

    asset = Asset()
    asset.joint_names = list(joint_names)
    asset.body_names = list(body_names)
    asset.cfg = Cfg()
    asset.cfg.joint_symmetry_mapping = joint_map
    asset.cfg.spatial_symmetry_mapping = spatial_map

    ids = torch.zeros(len(joint_names), dtype=torch.long)
    signs = torch.zeros(len(joint_names), dtype=torch.float32)
    for i, joint_name in enumerate(joint_names):
        sign, mirrored_name = joint_map[joint_name]
        ids[i] = joint_names.index(mirrored_name)
        signs[i] = sign
    joint_sym = SymmetryTransform(ids, signs)
    pos_sym = cartesian_space_symmetry(asset, asset.body_names, sign=(1, -1, 1))
    rot_sym = cartesian_space_symmetry(asset, asset.body_names, sign=(1, -1, 1, -1, 1, -1))

    t = 61
    root_pos = torch.zeros(1, t, 3)
    root_quat = torch.zeros(1, t, 4)
    root_quat[..., 0] = 1.0
    phase_t = torch.linspace(0.0, 2.0 * math.pi, t)
    joint_pos = torch.zeros(1, t, len(joint_names))
    for i, joint_name in enumerate(joint_names):
        joint = model.joint(joint_name)
        lo, hi = map(float, joint.range)
        mid = 0.5 * (lo + hi)
        amp = 0.2 * (hi - lo)
        joint_pos[0, :, i] = mid + amp * torch.sin((1.0 + 0.09 * (i % 5)) * phase_t + 0.23 * i)

    full = helper.expand_minimal_motion(
        MotionMinimalData(root_pos_w=root_pos, root_quat_w=root_quat, joint_pos=joint_pos),
        fps=50.0,
    )
    full_sym = helper.expand_minimal_motion(
        MotionMinimalData(root_pos_w=root_pos, root_quat_w=root_quat, joint_pos=joint_sym(joint_pos)),
        fps=50.0,
    )
    rot = quat_to_rot6d(full.body_quat_b).reshape(1, t, len(body_names), 6)
    rot_sym_fk = quat_to_rot6d(full_sym.body_quat_b).reshape(1, t, len(body_names), 6)

    body_pos_err = float(
        (
            full_sym.body_pos_b
            - pos_sym(full.body_pos_b.reshape(1, t, -1)).reshape_as(full.body_pos_b)
        )
        .abs()
        .max()
        .item()
    )
    body_quat_err = float(
        (rot_sym_fk - rot_sym(rot.reshape(1, t, -1)).reshape_as(rot))
        .abs()
        .max()
        .item()
    )
    return body_pos_err, body_quat_err


def generate_auto_symmetry(
    *,
    xml_path: Path,
    spec_fn: Callable[[], mujoco.MjSpec],
    body_pos_tol: float,
    body_quat_tol: float,
    fk_body_pos_tol: float,
    fk_body_quat_tol: float,
    name_swap_rules: NameSwapRules = DEFAULT_NAME_SWAP_RULES,
    validation_body_patterns: Sequence[str] | None = None,
) -> tuple[dict[str, tuple[int, str]], dict[str, str]]:
    """Generate and physically validate joint/body sagittal symmetry maps."""
    joint_axes, joint_order, body_names = parse_xml_symmetry_info(xml_path)
    _validate_name_swap_involution((*joint_order, *body_names), name_swap_rules)
    joint_map = build_auto_joint_symmetry_map(joint_axes, name_swap_rules)
    spatial_map = build_auto_spatial_symmetry_map(body_names, name_swap_rules)

    validation_body_names = _select_validation_bodies(
        body_names, validation_body_patterns
    )
    validation_body_set = set(validation_body_names)
    validation_spatial_map = {
        name: mirrored_name
        for name, mirrored_name in spatial_map.items()
        if name in validation_body_set and mirrored_name in validation_body_set
    }
    if len(validation_spatial_map) != len(validation_body_names):
        missing = validation_body_set.difference(validation_spatial_map)
        raise ValueError(f"Symmetry validation set is missing mirrored bodies: {missing}")

    spec = spec_fn()
    model = spec.compile()
    helper = _build_fk_helper(
        model=model,
        dataset_joint_names=joint_order,
        output_body_names=validation_body_names,
    )
    body_pos_err, body_quat_err = _validate_spatial_structure(
        model, validation_spatial_map
    )
    fk_body_pos_err, fk_body_quat_err = _validate_fk_pose_symmetry(
        model=model,
        helper=helper,
        joint_names=joint_order,
        body_names=validation_body_names,
        joint_map=joint_map,
        spatial_map=validation_spatial_map,
    )
    report = (
        f"[auto_symmetry] {xml_path.name}: "
        f"body_pos_err={body_pos_err:.6g} (tol={body_pos_tol:.6g}), "
        f"body_quat_err={body_quat_err:.6g} (tol={body_quat_tol:.6g}), "
        f"fk_body_pos_err={fk_body_pos_err:.6g} (tol={fk_body_pos_tol:.6g}), "
        f"fk_body_quat_err={fk_body_quat_err:.6g} (tol={fk_body_quat_tol:.6g})"
    )
    print(report)
    if (
        body_pos_err > body_pos_tol
        or body_quat_err > body_quat_tol
        or fk_body_pos_err > fk_body_pos_tol
        or fk_body_quat_err > fk_body_quat_tol
    ):
        raise ValueError(
            f"Auto symmetry validation failed for {xml_path.name}: "
            f"body_pos_err={body_pos_err:.6g}, "
            f"body_quat_err={body_quat_err:.6g}, "
            f"fk_body_pos_err={fk_body_pos_err:.6g}, "
            f"fk_body_quat_err={fk_body_quat_err:.6g}"
        )
    return joint_map, spatial_map
