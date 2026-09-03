"""Run a bounded, headless Roban tracking-environment smoke test."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = PROJECT_ROOT / "cfg"
PROJECT_DATA_ROOT = PROJECT_ROOT / "dataset"
EXPECTED_CONTACT_BODIES = ("leg_l6_link", "leg_r6_link")
MIB = 1024 * 1024

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_dataset_path(raw_path: Path) -> tuple[Path, str]:
    data_root = PROJECT_DATA_ROOT.resolve()
    mem_root = Path(os.environ.get("MEMPATH", PROJECT_DATA_ROOT)).resolve()
    if mem_root != data_root:
        raise ValueError(f"MEMPATH must resolve to {data_root}, got {mem_root}")
    candidate = raw_path if raw_path.is_absolute() else mem_root / raw_path
    resolved = candidate.resolve()
    if resolved == mem_root or not _is_relative_to(resolved, mem_root):
        raise ValueError(
            f"dataset path must be a child of {mem_root}: {raw_path} resolves to {resolved}"
        )
    required = ("_tensordict", "meta_motion.json", "id_label.json")
    missing = [name for name in required if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(f"dataset is incomplete at {resolved}; missing={missing}")
    return resolved, resolved.relative_to(mem_root).as_posix()


def _parse_cuda_device(raw_device: str) -> tuple[str, int]:
    match = re.fullmatch(r"cuda(?::([0-9]+))?", raw_device.strip().lower())
    if match is None:
        raise ValueError("Roban MJLab smoke currently requires --device cuda:N")
    index = int(match.group(1) or 0)
    canonical = f"cuda:{index}"
    existing_rank = os.environ.get("LOCAL_RANK")
    if existing_rank is not None and int(existing_rank) != index:
        raise ValueError(
            f"LOCAL_RANK={existing_rank} conflicts with requested device {canonical}"
        )
    os.environ["LOCAL_RANK"] = str(index)
    return canonical, index


def _compose_cfg(num_envs: int, dataset_mem_path: str, seed: int):
    import active_adaptation.learning  # noqa: F401 - registers Hydra algo configs
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    dataset_override = (
        "task.profile.command.dataset.groups="
        f"[{{name:env_smoke,teacher_mem_path:{json.dumps(dataset_mem_path)},weight:1.0}}]"
    )
    overrides = [
        "task=tracking",
        "task/profile=Roban_tracking",
        f"task.num_envs={num_envs}",
        dataset_override,
        f"seed={seed}",
        "headless=true",
        "app.headless=true",
        "app.enable_cameras=false",
        "eval_render=false",
        "checkpoint_path=null",
        "vecnorm=null",
        "wandb.mode=disabled",
        "train_record.enabled=false",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CFG_DIR),
        job_name="roban_env_smoke",
    ):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg, overrides


def _assert_finite(torch, label: str, value, checks: dict[str, bool]) -> None:
    if value is None:
        raise AssertionError(f"{label} is unavailable")
    finite = bool(torch.isfinite(value).all().item())
    checks[label] = checks.get(label, True) and finite
    if not finite:
        bad_count = int((~torch.isfinite(value)).sum().item())
        raise FloatingPointError(f"{label} contains {bad_count} NaN/Inf values")


def _assert_observations_finite(
    torch,
    tensordict,
    observation_keys: Iterable[str],
    prefix: str,
    checks: dict[str, bool],
) -> None:
    for key in observation_keys:
        value = tensordict.get(key, None)
        _assert_finite(torch, f"{prefix}.{key}", value, checks)


def _runtime_regex_audit(cfg, robot) -> dict[str, list[str]]:
    joint_names = tuple(robot.joint_names)
    body_names = tuple(robot.body_names)
    matched: dict[str, list[str]] = {}

    def check(label: str, pattern: str, names: tuple[str, ...]) -> None:
        values = [name for name in names if re.fullmatch(str(pattern), name)]
        matched[label] = values
        if not values:
            raise AssertionError(f"runtime regex matched nothing: {label}={pattern!r}")

    for key, pattern in cfg.task.profile.body.items():
        if key.endswith("_pattern"):
            check(f"profile.body.{key}", pattern, body_names)
    for index, pattern in enumerate(cfg.task.command.required_motion_body_patterns):
        check(f"command.required_motion_body_patterns[{index}]", pattern, body_names)
    for index, pattern in enumerate(cfg.task.command.keypoint_patterns):
        check(f"command.keypoint_patterns[{index}]", pattern, body_names)
    for index, pattern in enumerate(cfg.task.command.feet_patterns):
        check(f"command.feet_patterns[{index}]", pattern, body_names)

    body_z_patterns = cfg.task.termination.body_z_termination.body_z_terminate_patterns
    for index, pattern in enumerate(body_z_patterns):
        check(f"termination.body_z[{index}]", pattern, body_names)

    for index, pattern in enumerate(cfg.task.randomization.perturb_body_com.body_names):
        check(f"randomization.perturb_body_com[{index}]", pattern, body_names)
    check(
        "randomization.perturb_body_wrench",
        cfg.task.randomization.perturb_body_wrench.body_name,
        body_names,
    )
    check(
        "randomization.perturb_body_materials",
        cfg.task.randomization.perturb_body_materials.body_names,
        body_names,
    )

    action_counts = {name: 0 for name in joint_names}
    for index, pattern in enumerate(cfg.task.action.action_scaling.keys()):
        check(f"action.action_scaling[{index}]", pattern, joint_names)
        for name in matched[f"action.action_scaling[{index}]"]:
            action_counts[name] += 1
    invalid_counts = {name: count for name, count in action_counts.items() if count != 1}
    if invalid_counts:
        raise AssertionError(f"action regex coverage is not exactly once: {invalid_counts}")
    return matched


def _unique_contact_primary_names(contact_sensor) -> tuple[str, ...]:
    names: list[str] = []
    for slot in contact_sensor._slots:
        if slot.primary_name not in names:
            names.append(slot.primary_name)
    return tuple(names)


def _gpu_memory(torch, device: str) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_mib": int(free_bytes // MIB),
        "used_mib": int((total_bytes - free_bytes) // MIB),
        "torch_allocated_mib": int(torch.cuda.memory_allocated(device) // MIB),
        "torch_reserved_mib": int(torch.cuda.memory_reserved(device) // MIB),
    }


def run(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    device, device_index = _parse_cuda_device(args.device)
    dataset_path, dataset_mem_path = _resolve_dataset_path(args.dataset_path)

    import numpy as np
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested {device}, but only {torch.cuda.device_count()} visible CUDA device(s)"
        )
    torch.cuda.set_device(device_index)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    cfg, overrides = _compose_cfg(args.num_envs, dataset_mem_path, args.seed)
    print(f"hydra_overrides={overrides}", flush=True)
    print(
        "disabled_outputs="
        f"wandb:{cfg.wandb.mode},checkpoint:{cfg.checkpoint_path},"
        f"video:{cfg.eval_render},renderer:{cfg.app.enable_cameras},headless:{cfg.app.headless}",
        flush=True,
    )
    print(f"dataset_requested={dataset_path}", flush=True)
    print(f"gpu_memory_before_env={_gpu_memory(torch, device)}", flush=True)

    from active_adaptation.assets.roban import ROBAN_JOINT_ORDER
    from scripts.utils.helpers import make_env_policy

    env = None
    policy = None
    finite_checks: dict[str, bool] = {}
    memory_samples: list[dict[str, int]] = []
    max_quat_norm_deviation = 0.0
    try:
        env, policy, _, _ = make_env_policy(cfg)
        base_env = env.base_env
        if str(base_env.device) != device:
            raise AssertionError(f"env device {base_env.device} != requested {device}")

        robot = base_env.scene["robot"]
        joint_order = tuple(base_env.action_manager.joint_names)
        if cfg.task.robot.name != "roban":
            raise AssertionError(f"robot name is {cfg.task.robot.name!r}, not 'roban'")
        if base_env.action_dim != 21:
            raise AssertionError(f"controlled joint count is {base_env.action_dim}, not 21")
        if joint_order != tuple(ROBAN_JOINT_ORDER):
            raise AssertionError(
                f"controlled joint order mismatch: runtime={joint_order}, expected={ROBAN_JOINT_ORDER}"
            )

        command_dataset = base_env.command_manager.dataset
        teacher_dataset = command_dataset.teacher_dataset
        if str(command_dataset.ds_device) != "cpu" or str(teacher_dataset.ds_device) != "cpu":
            raise AssertionError(
                "smoke dataset must remain on CPU: "
                f"cache.ds_device={command_dataset.ds_device}, "
                f"teacher.ds_device={teacher_dataset.ds_device}"
            )
        resolved_loaded_paths = tuple(
            (Path(path) if Path(path).is_absolute() else PROJECT_DATA_ROOT / path).resolve()
            for path in teacher_dataset.dataset_paths
        )
        if resolved_loaded_paths != (dataset_path,):
            raise AssertionError(
                f"loaded dataset paths {resolved_loaded_paths} != requested {(dataset_path,)}"
            )

        if abs(float(base_env.step_dt) - 0.02) > 1.0e-12:
            raise AssertionError(f"step_dt must be 0.02, got {base_env.step_dt}")
        if base_env.decimation != 4:
            raise AssertionError(f"decimation must be 4, got {base_env.decimation}")

        regex_matches = _runtime_regex_audit(cfg, robot)
        contact_sensor = base_env.scene["contact_forces"]
        contact_names = _unique_contact_primary_names(contact_sensor)
        if contact_names != EXPECTED_CONTACT_BODIES:
            raise AssertionError(
                f"contact sensor primaries {contact_names} != {EXPECTED_CONTACT_BODIES}"
            )

        dense_toe_reward = base_env.reward_groups["loco"].funcs[
            "feet_air_time_ref_dense"
        ]
        toe_parsed_count = int(dense_toe_reward.body_ids.numel())
        if dense_toe_reward.enabled:
            raise AssertionError("Roban feet_air_time_ref_dense must be disabled")
        if toe_parsed_count != 0 or dense_toe_reward.contact_cache is not None:
            raise AssertionError("disabled Roban toe reward parsed optional toe bodies")

        print("robot_name=roban", flush=True)
        print(f"controlled_joint_count={base_env.action_dim}", flush=True)
        print(f"joint_order={joint_order}", flush=True)
        print(
            f"dataset_loaded={resolved_loaded_paths} motions={teacher_dataset.num_motions} "
            f"dataset_device={teacher_dataset.ds_device}",
            flush=True,
        )
        print(
            f"step_dt={base_env.step_dt} physics_dt={base_env.physics_dt} "
            f"decimation={base_env.decimation}",
            flush=True,
        )
        print(
            f"contact_sensor_primaries={contact_names} "
            f"contact_pattern={contact_sensor.cfg.primary.pattern}",
            flush=True,
        )
        print(
            "toe_reward_enabled="
            f"{dense_toe_reward.enabled} toe_body_parsed_count={toe_parsed_count} "
            f"toe_contact_cache={dense_toe_reward.contact_cache}",
            flush=True,
        )
        print(
            f"runtime_regex_checks={len(regex_matches)} empty_matches=[]",
            flush=True,
        )

        carry = env.reset()
        observation_keys = tuple(base_env.observation_funcs.keys())
        _assert_observations_finite(
            torch, carry, observation_keys, "reset_observation", finite_checks
        )
        reset_policy_obs = carry["policy"]
        if reset_policy_obs.shape != (args.num_envs, 1313):
            raise AssertionError(
                f"reset policy observation shape {tuple(reset_policy_obs.shape)} != "
                f"{(args.num_envs, 1313)}"
            )

        _assert_finite(torch, "reset.qpos", base_env.sim.data.qpos, finite_checks)
        _assert_finite(torch, "reset.qvel", base_env.sim.data.qvel, finite_checks)
        reset_contact = contact_sensor.data
        _assert_finite(torch, "reset.contact_found", reset_contact.found, finite_checks)
        _assert_finite(torch, "reset.contact_force", reset_contact.force, finite_checks)

        student_td = carry.clone()
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            policy.adapt_module(student_td)
            continuous_latent = student_td["priv_pred"]
            policy.student_latent_quantizer(student_td)
            quantized_latent = student_td["student_priv_feature"]
            policy.actor_student(student_td)
        actor_input = student_td["_actor_inp"]
        student_action = student_td["action"]
        expected_shapes = {
            "observation": (args.num_envs, 1313),
            "continuous_latent": (args.num_envs, 40),
            "quantized_latent": (args.num_envs, 40),
            "actor_input": (args.num_envs, 1353),
            "action": (args.num_envs, 21),
        }
        actual_tensors = {
            "observation": reset_policy_obs,
            "continuous_latent": continuous_latent,
            "quantized_latent": quantized_latent,
            "actor_input": actor_input,
            "action": student_action,
        }
        for name, tensor in actual_tensors.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise AssertionError(
                    f"Student {name} shape {tuple(tensor.shape)} != {expected_shapes[name]}"
                )
            _assert_finite(torch, f"student_forward.{name}", tensor, finite_checks)
        print(
            "student_forward_shapes="
            f"obs:{tuple(reset_policy_obs.shape)},"
            f"adapt:{tuple(continuous_latent.shape)},"
            f"quantizer:{tuple(quantized_latent.shape)},"
            f"actor_input:{tuple(actor_input.shape)},"
            f"action:{tuple(student_action.shape)}",
            flush=True,
        )
        print(
            "random_initialized_student_forward_only=true policy_tracking_validated=false",
            flush=True,
        )

        zero_action = torch.zeros(
            args.num_envs, base_env.action_dim, device=device, dtype=torch.float32
        )
        memory_samples.append(_gpu_memory(torch, device))
        for step_index in range(args.steps):
            carry.set("action", zero_action)
            transition, carry = env.step_and_maybe_reset(carry)
            next_td = transition["next"]
            _assert_observations_finite(
                torch,
                next_td,
                observation_keys,
                "step_observation",
                finite_checks,
            )
            if next_td["policy"].shape[-1] != 1313:
                raise AssertionError(
                    f"step policy observation width {next_td['policy'].shape[-1]} != 1313"
                )
            if zero_action.shape[-1] != 21:
                raise AssertionError(f"step action width {zero_action.shape[-1]} != 21")

            _assert_finite(torch, "step.qpos", base_env.sim.data.qpos, finite_checks)
            _assert_finite(torch, "step.qvel", base_env.sim.data.qvel, finite_checks)
            _assert_finite(
                torch, "step.qfrc_actuator", robot.data.qfrc_actuator, finite_checks
            )
            _assert_finite(
                torch, "step.actuator_force", robot.data.actuator_force, finite_checks
            )
            _assert_finite(torch, "step.reward", next_td["reward"], finite_checks)

            contact_data = contact_sensor.data
            _assert_finite(
                torch, "step.contact_found", contact_data.found, finite_checks
            )
            _assert_finite(
                torch, "step.contact_force", contact_data.force, finite_checks
            )
            if contact_data.force_history is not None:
                _assert_finite(
                    torch,
                    "step.contact_force_history",
                    contact_data.force_history,
                    finite_checks,
                )

            root_quat = robot.data.root_link_quat_w
            _assert_finite(torch, "step.root_quaternion", root_quat, finite_checks)
            quat_deviation = float((root_quat.norm(dim=-1) - 1.0).abs().max().item())
            max_quat_norm_deviation = max(max_quat_norm_deviation, quat_deviation)
            if quat_deviation > 1.0e-3:
                raise FloatingPointError(
                    f"root quaternion norm deviation {quat_deviation:.8g} exceeds 1e-3"
                )

            memory_samples.append(_gpu_memory(torch, device))
            if step_index == 0 or step_index + 1 == args.steps:
                print(
                    f"step={step_index + 1}/{args.steps} "
                    f"obs_shape={tuple(next_td['policy'].shape)} "
                    f"reward_shape={tuple(next_td['reward'].shape)} "
                    f"gpu_memory={memory_samples[-1]}",
                    flush=True,
                )

        warmup_sample = memory_samples[min(5, len(memory_samples) - 1)]
        final_sample = memory_samples[-1]
        steady_growth_mib = final_sample["used_mib"] - warmup_sample["used_mib"]
        if steady_growth_mib > 512:
            raise RuntimeError(
                f"GPU memory grew by {steady_growth_mib} MiB after warmup (>512 MiB)"
            )
        print(f"reset_observation_shape={tuple(reset_policy_obs.shape)} finite=true", flush=True)
        print(
            f"step_observation_shape={tuple(carry['policy'].shape)} finite=true",
            flush=True,
        )
        print(f"finite_checks={finite_checks}", flush=True)
        print(
            f"root_quaternion_max_norm_deviation={max_quat_norm_deviation:.8g}",
            flush=True,
        )
        print(
            f"gpu_memory_after_warmup={warmup_sample} "
            f"gpu_memory_final={final_sample} steady_growth_mib={steady_growth_mib} "
            f"torch_peak_allocated_mib={int(torch.cuda.max_memory_allocated(device) // MIB)}",
            flush=True,
        )
        print("SMOKE_RESULT=PASS", flush=True)
    finally:
        if env is not None:
            env.close()
        del policy
        del env
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            print(f"gpu_memory_after_close={_gpu_memory(torch, device)}", flush=True)


def main() -> int:
    args = _parser().parse_args()
    try:
        run(args)
    except BaseException:
        print("SMOKE_RESULT=FAIL", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
