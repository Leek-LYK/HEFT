from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from active_adaptation.assets.roban import ROBAN_JOINT_ORDER, ROBAN_XML
from active_adaptation.utils.roban_ominisoma import SOURCE_BODY_NAMES, SOURCE_JOINT_NAMES


def basic_source_arrays(frame_count: int = 160) -> dict[str, np.ndarray]:
    root_pos = np.zeros((frame_count, 3), dtype=np.float32)
    root_pos[:, 2] = 0.77
    body_pos = np.repeat(root_pos[:, None, :], len(SOURCE_BODY_NAMES), axis=1)
    body_pos[..., 2] = 0.5
    body_pos[:, 0] = root_pos
    body_pos[:, SOURCE_BODY_NAMES.index("leg_l6_link"), 2] = 0.08
    body_pos[:, SOURCE_BODY_NAMES.index("leg_r6_link"), 2] = 0.08
    body_quat = np.zeros((frame_count, len(SOURCE_BODY_NAMES), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    joint_pos = np.repeat(
        np.arange(len(SOURCE_JOINT_NAMES), dtype=np.float32)[None, :],
        frame_count,
        axis=0,
    )
    joint_vel = joint_pos + 100.0
    return {
        "body_ang_vel_w": np.zeros_like(body_pos),
        "body_lin_vel_w": np.zeros_like(body_pos),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "fps": np.asarray([50], dtype=np.int64),
    }


def write_source_npz(
    path: Path,
    *,
    arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **(basic_source_arrays() if arrays is None else arrays))
    return path


def write_fk_source_npz(path: Path, frame_count: int = 160) -> tuple[Path, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(ROBAN_XML))
    data = mujoco.MjData(model)
    canonical = np.zeros((frame_count, len(ROBAN_JOINT_ORDER)), dtype=np.float32)
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count, dtype=np.float32)
    for index in range(len(ROBAN_JOINT_ORDER)):
        canonical[:, index] = 0.04 * np.sin(phase + index * 0.1)

    root_pos = np.zeros((frame_count, 3), dtype=np.float32)
    root_pos[:, 2] = 0.77
    root_quat = np.zeros((frame_count, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    body_pos = np.empty((frame_count, len(SOURCE_BODY_NAMES), 3), dtype=np.float32)
    body_quat = np.empty((frame_count, len(SOURCE_BODY_NAMES), 4), dtype=np.float32)
    model_body_names = {model.body(index).name for index in range(1, model.nbody)}

    for frame in range(frame_count):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        for name, value in zip(ROBAN_JOINT_ORDER, canonical[frame]):
            data.qpos[model.joint(name).qposadr] = value
        mujoco.mj_forward(model, data)
        for body_index, body_name in enumerate(SOURCE_BODY_NAMES):
            if body_name in model_body_names:
                body_id = model.body(body_name).id
                body_pos[frame, body_index] = data.xpos[body_id]
                body_quat[frame, body_index] = data.xquat[body_id]
            else:
                # Legacy-only hand/camera links are quality-audit inputs and do
                # not participate in the current-asset FK reconstruction.
                body_pos[frame, body_index] = data.xpos[model.body("base_link").id]
                body_quat[frame, body_index] = data.xquat[model.body("base_link").id]

    source_joint_pos = np.empty_like(canonical)
    for source_index, source_name in enumerate(SOURCE_JOINT_NAMES):
        canonical_index = ROBAN_JOINT_ORDER.index(f"{source_name}_joint")
        source_joint_pos[:, source_index] = canonical[:, canonical_index]
    joint_vel = np.gradient(source_joint_pos, axis=0) * 50.0
    body_lin_vel = np.gradient(body_pos, axis=0) * 50.0
    arrays = {
        "body_ang_vel_w": np.zeros_like(body_pos),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "joint_pos": source_joint_pos,
        "joint_vel": joint_vel.astype(np.float32),
        "fps": np.asarray([50], dtype=np.int64),
    }
    return write_source_npz(path, arrays=arrays), canonical
