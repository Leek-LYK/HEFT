"""Deterministic, optimizer-free evaluation for a Roban Teacher checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Allow direct execution from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from active_adaptation.utils.roban_ominisoma import sha256_file
from scripts.utils.helpers import make_env_policy


EXPECTED_OBSERVATION_DIM = 1313
EXPECTED_LATENT_DIM = 40
EXPECTED_ACTION_DIM = 21
MIB = 1024 * 1024
AGE_BUCKETS = (
    ("0-49", 0, 50),
    ("50-99", 50, 100),
    ("100-199", 100, 200),
    ("200-399", 200, 400),
    ("400+", 400, None),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_samples(values: torch.Tensor) -> dict[str, float | int | None]:
    """Summarize a finite, one-dimensional CPU sample tensor."""
    values = values.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if values.numel() == 0:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError("metric samples contain NaN/Inf")
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def summarize_root_errors(
    xy: torch.Tensor, z: torch.Tensor, orientation: torch.Tensor
) -> dict[str, dict[str, float | int | None]]:
    return {
        "root_xy_position_error_m": summarize_samples(xy),
        "root_z_error_m": summarize_samples(z),
        "root_orientation_geodesic_error_rad": summarize_samples(orientation),
    }


def _root_tracking_errors(base_env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    command = base_env.command_manager
    current_pos = base_env.scene["robot"].data.root_link_pos_w
    target_pos = command.reward_root_pos_w
    delta = current_pos - target_pos
    xy = torch.linalg.vector_norm(delta[:, :2], dim=-1)
    z = delta[:, 2].abs()

    current_quat = base_env.scene["robot"].data.root_link_quat_w
    target_quat = command.reward_root_quat_w
    current_quat = current_quat / torch.linalg.vector_norm(
        current_quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    target_quat = target_quat / torch.linalg.vector_norm(
        target_quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    dot = (current_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    orientation = 2.0 * torch.acos(dot)
    return xy, z, orientation


class EpisodeAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.completed = torch.zeros((), dtype=torch.long, device=device)
        self.terminated = torch.zeros((), dtype=torch.long, device=device)
        self.truncated = torch.zeros((), dtype=torch.long, device=device)
        self.immediate = torch.zeros((), dtype=torch.long, device=device)
        self.stat_sums: dict[str, torch.Tensor] = {}

    def add(self, transition) -> None:
        done = transition["next", "done"].reshape(-1)
        if not bool(done.any().item()):
            return
        terminated = transition["next", "terminated"].reshape(-1)
        truncated = transition["next", "truncated"].reshape(-1)
        stats = transition["next", "stats"]
        episode_len = stats["episode_len"].reshape(-1)
        self.completed.add_(done.sum())
        self.terminated.add_((done & terminated).sum())
        self.truncated.add_((done & truncated).sum())
        self.immediate.add_((done & terminated & (episode_len <= 10)).sum())
        for key, tensor in stats.items(True, True):
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != done.shape[0]:
                continue
            name = "/".join(key) if isinstance(key, tuple) else str(key)
            value = tensor[done].detach().double().sum()
            if name not in self.stat_sums:
                self.stat_sums[name] = torch.zeros_like(value)
            self.stat_sums[name].add_(value)

    def summary(self) -> dict[str, Any]:
        completed = int(self.completed.item())
        terminated = int(self.terminated.item())
        truncated = int(self.truncated.item())
        immediate = int(self.immediate.item())
        sums = {name: float(value.item()) for name, value in sorted(self.stat_sums.items())}
        means = {
            name: value / completed for name, value in sums.items()
        } if completed else {}
        episode_length_sum = sums.get("episode_len", 0.0)
        per_step = {
            name: value / episode_length_sum
            for name, value in sums.items()
            if name != "episode_len" and episode_length_sum > 0.0
        }
        components = {
            name: {
                "per_episode_cumulative": means.get(name),
                "per_step": per_step.get(name),
                "sum": value,
            }
            for name, value in sums.items()
            if name != "episode_len" and not name.startswith("termination/")
        }
        tracking_return = means.get("tracking/return")
        loco_return = means.get("loco/return")
        mean_reward = (
            None
            if tracking_return is None and loco_return is None
            else float((tracking_return or 0.0) + (loco_return or 0.0))
        )
        body_z_count = sums.get("termination/body_z_termination", 0.0)
        return {
            "completed_episodes": completed,
            "terminated_episodes": terminated,
            "truncated_episodes": truncated,
            "mean_episode_reward": mean_reward,
            "mean_episode_length": means.get("episode_len"),
            "body_z_termination_count": int(round(body_z_count)),
            "body_z_termination_ratio": (
                body_z_count / completed if completed else None
            ),
            "termination_ratio": terminated / completed if completed else None,
            "terminations_le_10_steps": immediate,
            "termination_le_10_steps_ratio": (
                immediate / completed if completed else None
            ),
            "tracking_return": tracking_return,
            "loco_return": loco_return,
            "episode_length_sum": episode_length_sum,
            "reward_components": components,
            "episode_stat_sums": sums,
        }


class RootErrorAccumulator:
    def __init__(self) -> None:
        self.xy: list[torch.Tensor] = []
        self.z: list[torch.Tensor] = []
        self.orientation: list[torch.Tensor] = []
        self.age: list[torch.Tensor] = []

    def add(
        self,
        xy: torch.Tensor,
        z: torch.Tensor,
        orientation: torch.Tensor,
        age: torch.Tensor,
    ) -> None:
        self.xy.append(xy.detach().reshape(-1).to(device="cpu", dtype=torch.float32))
        self.z.append(z.detach().reshape(-1).to(device="cpu", dtype=torch.float32))
        self.orientation.append(
            orientation.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        )
        self.age.append(age.detach().reshape(-1).to(device="cpu", dtype=torch.long))

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.xy:
            return tuple(torch.empty(0) for _ in range(4))  # type: ignore[return-value]
        return (
            torch.cat(self.xy),
            torch.cat(self.z),
            torch.cat(self.orientation),
            torch.cat(self.age),
        )

    def summary(self) -> dict[str, Any]:
        xy, z, orientation, age = self.tensors()
        result = summarize_root_errors(xy, z, orientation)
        by_age = {}
        for label, lower, upper in AGE_BUCKETS:
            mask = age >= lower
            if upper is not None:
                mask &= age < upper
            by_age[label] = summarize_root_errors(
                xy[mask], z[mask], orientation[mask]
            )
        result["by_episode_age_steps"] = by_age
        return result


def _parameter_snapshot(policy) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
    }


def _parameter_delta(
    before: Mapping[str, torch.Tensor], policy
) -> dict[str, Any]:
    after = dict(policy.named_parameters())
    changed = []
    maximum = 0.0
    for name, old in before.items():
        new = after[name].detach().cpu()
        delta = float((new - old).abs().max().item()) if new.numel() else 0.0
        maximum = max(maximum, delta)
        if delta != 0.0:
            changed.append(name)
    return {
        "changed": bool(changed),
        "changed_parameter_count": len(changed),
        "changed_parameters": changed,
        "max_abs_delta": maximum,
    }


def _tensor_state_snapshot(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        str(name): value.detach().cpu().clone()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def _tensor_state_delta(
    before: Mapping[str, torch.Tensor], state: Mapping[str, Any]
) -> dict[str, Any]:
    after = {
        str(name): value.detach().cpu()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    missing = sorted(set(before) - set(after))
    unexpected = sorted(set(after) - set(before))
    changed = []
    maximum = 0.0
    for name in sorted(set(before) & set(after)):
        delta = float((after[name] - before[name]).abs().max().item())
        maximum = max(maximum, delta)
        if delta != 0.0:
            changed.append(name)
    return {
        "changed": bool(changed or missing or unexpected),
        "changed_keys": changed,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "max_abs_delta": maximum,
    }


def _memory_sample(device: torch.device, label: str) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    current_rss_mib = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                current_rss_mib = float(line.split()[1]) / 1024.0
                break
    except OSError:
        pass
    return {
        "label": label,
        "gpu_device_used_mib": int((total_bytes - free_bytes) // MIB),
        "gpu_torch_allocated_mib": int(torch.cuda.memory_allocated(device) // MIB),
        "gpu_torch_reserved_mib": int(torch.cuda.memory_reserved(device) // MIB),
        "cpu_current_rss_mib": current_rss_mib,
        "cpu_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        / 1024.0,
    }


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        count = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"{name} contains {count} NaN/Inf value(s)")


def _prepare_config(args: argparse.Namespace, output_dir: Path):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "cfg" not in checkpoint:
        raise KeyError("checkpoint has no embedded cfg")
    cfg = OmegaConf.create(
        copy.deepcopy(OmegaConf.to_container(checkpoint["cfg"], resolve=False))
    )
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.seed = int(args.seed)
    cfg.vecnorm = "eval"
    cfg.task.num_envs = int(args.num_envs)
    cfg.task.profile.command.dataset.groups = [
        {
            "name": "medium_teacher_v1",
            "teacher_mem_path": str(args.dataset),
            "weight": 1.0,
        }
    ]
    cfg.task.command.dataset.groups = cfg.task.profile.command.dataset.groups
    cfg.headless = True
    cfg.eval_render = False
    cfg.app.headless = True
    cfg.app.enable_cameras = False
    cfg.wandb.mode = "disabled"
    cfg.train_record.enabled = False
    if cfg.get("pilot") is not None:
        cfg.pilot.enabled = False
    cfg.task.recover_nonfinite_state = True
    cfg.task.termination.nonfinite_state = {}
    cfg.task.nonfinite_diagnostics = {
        "enabled": True,
        "output_dir": str(output_dir),
        "detailed_event_limit": 200,
        "rate_min_policy_transitions": 100_000,
        "max_event_rate": 1.0e-4,
        "max_motion_frame_repeats": 10,
        "max_events_per_iteration": 20,
        "initial_summary_path": None,
        "initial_events_path": None,
    }
    OmegaConf.resolve(cfg)
    return cfg


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError(
            "Roban Teacher evaluation requires CUDA_VISIBLE_DEVICES=3"
        )
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("Roban Teacher evaluation is single-GPU only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.rollout_iterations <= 0 or args.num_envs <= 0:
        raise ValueError("rollout iterations and num envs must be positive")

    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    cfg = _prepare_config(args, output_dir)
    if int(cfg.algo.train_every) != int(args.rollout_horizon):
        raise AssertionError(
            f"formal rollout horizon {cfg.algo.train_every} != requested {args.rollout_horizon}"
        )
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    env = None
    result: dict[str, Any] = {
        "status": "RUNNING",
        "argv": list(sys.argv),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "evaluation.json", result)
    try:
        torch.cuda.reset_peak_memory_stats()
        env, policy, vecnorm, _ = make_env_policy(cfg)
        base_env = env.base_env
        device = torch.device(base_env.device)
        if device != torch.device("cuda:0"):
            raise AssertionError(f"expected logical cuda:0, got {device}")
        dataset = base_env.command_manager.dataset
        teacher_dataset = dataset.teacher_dataset
        if str(dataset.ds_device) != "cpu" or str(teacher_dataset.ds_device) != "cpu":
            raise AssertionError("dataset and memdataset cache must remain on CPU")
        if int(base_env.num_envs) != int(args.num_envs):
            raise AssertionError("environment count mismatch")
        if int(base_env.action_dim) != EXPECTED_ACTION_DIM:
            raise AssertionError("action contract changed")
        if int(cfg.algo.latent_dim) != EXPECTED_LATENT_DIM:
            raise AssertionError("Teacher latent contract changed")

        schedule_progress = float(getattr(policy, "progress", 0.0))
        policy.step_schedule(schedule_progress, -1)
        env.step_schedule(schedule_progress, -1)
        policy.eval()
        env.eval()
        env.set_seed(int(args.seed))

        optimizer_step_attempts = defaultdict(int)
        for name in ("teacher", "student", "critic", "estimator"):
            optimizer = getattr(policy, f"opt_{name}")

            def forbidden_step(*_args, _name=name, **_kwargs):
                optimizer_step_attempts[_name] += 1
                raise AssertionError(f"optimizer.step attempted during evaluation: {_name}")

            optimizer.step = forbidden_step

        parameters_before = _parameter_snapshot(policy)
        vecnorm_before = _tensor_state_snapshot(vecnorm.state_dict())
        rollout_policy = policy.get_rollout_policy("eval")
        carry = env.reset()
        if tuple(carry["policy"].shape) != (
            int(args.num_envs),
            EXPECTED_OBSERVATION_DIM,
        ):
            raise AssertionError(
                f"policy observation shape changed: {tuple(carry['policy'].shape)}"
            )
        _assert_finite("reset.policy", carry["policy"])

        initial_motion_ids = dataset._motion_ids_A.detach().cpu()
        motion_count = len(dataset.motion_labels)
        motion_transition_counts = torch.zeros(
            motion_count, dtype=torch.long, device=device
        )
        all_episodes = EpisodeAccumulator(device)
        all_root_errors = RootErrorAccumulator()
        memory_samples = [_memory_sample(device, "after_initialization")]
        rollout_rows = []
        rollout_path = output_dir / "rollout_metrics.jsonl"
        action_mean_max_abs_delta = 0.0
        total_environment_transitions = 0
        evaluation_start = time.perf_counter()

        with rollout_path.open("w", encoding="utf-8") as rollout_stream:
            with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
                for rollout_index in range(1, int(args.rollout_iterations) + 1):
                    rollout_start = time.perf_counter()
                    rollout_episode = EpisodeAccumulator(device)
                    rollout_root = RootErrorAccumulator()
                    rollout_motion_seen = torch.zeros(
                        motion_count, dtype=torch.bool, device=device
                    )
                    recorder = base_env.nonfinite_diagnostics
                    events_before = 0 if recorder is None else int(recorder.total_events)
                    for rollout_step in range(int(args.rollout_horizon)):
                        global_step = (
                            (rollout_index - 1) * int(args.rollout_horizon)
                            + rollout_step
                        )
                        if recorder is not None:
                            base_env.set_nonfinite_diagnostic_context(
                                policy_iteration=rollout_index,
                                global_policy_frame=(global_step + 1)
                                * int(args.num_envs),
                                global_physics_tick_base=global_step
                                * int(base_env.decimation),
                            )
                        torch.compiler.cudagraph_mark_step_begin()
                        carry = rollout_policy(carry)
                        _assert_finite("action", carry["action"])
                        _assert_finite("action_mean", carry["loc"])
                        action_delta = float(
                            (carry["action"] - carry["loc"]).abs().max().item()
                        )
                        action_mean_max_abs_delta = max(
                            action_mean_max_abs_delta, action_delta
                        )
                        if action_delta != 0.0:
                            raise AssertionError(
                                f"MODE action differs from mean by {action_delta}"
                            )

                        # Split TorchRL's step_and_maybe_reset helper so root
                        # errors are captured from the terminal state before a
                        # completed environment is reset.
                        transition = env.step(carry)
                        _assert_finite("reward", transition["next", "reward"])
                        _assert_finite("next.policy", transition["next", "policy"])
                        asset_data = base_env.scene["robot"].data
                        for name, tensor in (
                            ("qpos", base_env.sim.data.qpos),
                            ("qvel", base_env.sim.data.qvel),
                            ("torque", asset_data.qfrc_actuator),
                            ("joint_target", asset_data.joint_pos_target),
                            ("ctrl", base_env.sim.data.ctrl),
                        ):
                            _assert_finite(name, tensor)

                        all_episodes.add(transition)
                        rollout_episode.add(transition)
                        xy, z, orientation = _root_tracking_errors(base_env)
                        age = transition[
                            "next", "stats", "episode_len"
                        ].reshape(-1)
                        all_root_errors.add(xy, z, orientation, age)
                        rollout_root.add(xy, z, orientation, age)

                        motion_ids = dataset._motion_ids_A.long()
                        ones = torch.ones_like(motion_ids, dtype=torch.long)
                        motion_transition_counts.index_add_(0, motion_ids, ones)
                        rollout_motion_seen[motion_ids] = True
                        total_environment_transitions += int(args.num_envs)
                        carry = env._step_mdp(transition)
                        carry = env.maybe_reset(carry)

                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - rollout_start
                    memory = _memory_sample(device, f"rollout_{rollout_index}")
                    memory_samples.append(memory)
                    events_after = 0 if recorder is None else int(recorder.total_events)
                    row = {
                        "rollout_index": rollout_index,
                        "environment_transitions": int(args.num_envs)
                        * int(args.rollout_horizon),
                        "elapsed_s": elapsed,
                        "environment_transitions_per_second": (
                            int(args.num_envs) * int(args.rollout_horizon) / elapsed
                        ),
                        "policy_steps_per_second": int(args.rollout_horizon) / elapsed,
                        "completed_episode_metrics": rollout_episode.summary(),
                        "root_tracking_error": rollout_root.summary(),
                        "sampled_unique_motions": int(
                            rollout_motion_seen.sum().item()
                        ),
                        "nonfinite_events": events_after - events_before,
                        "memory": memory,
                    }
                    rollout_rows.append(row)
                    rollout_stream.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                    rollout_stream.flush()
                    if rollout_index == 1 or rollout_index % 10 == 0:
                        print(
                            "TEACHER_EVAL_UPDATE "
                            f"rollout={rollout_index}/{args.rollout_iterations} "
                            f"fps={row['environment_transitions_per_second']:.1f} "
                            f"episodes={row['completed_episode_metrics']['completed_episodes']} "
                            f"nonfinite={row['nonfinite_events']}",
                            flush=True,
                        )

        torch.cuda.synchronize(device)
        elapsed_total = time.perf_counter() - evaluation_start
        parameter_delta = _parameter_delta(parameters_before, policy)
        vecnorm_delta = _tensor_state_delta(vecnorm_before, vecnorm.state_dict())
        if parameter_delta["changed"]:
            raise AssertionError(f"policy parameters changed during eval: {parameter_delta}")
        if vecnorm_delta["changed"]:
            raise AssertionError(f"VecNorm changed during eval: {vecnorm_delta}")
        if any(optimizer_step_attempts.values()):
            raise AssertionError(
                f"optimizer step attempts during eval: {dict(optimizer_step_attempts)}"
            )

        nonzero_motion_ids = motion_transition_counts.nonzero().reshape(-1).cpu()
        motion_distribution = []
        for motion_id in nonzero_motion_ids.tolist():
            motion_distribution.append(
                {
                    "motion_id": int(motion_id),
                    "transition_count": int(
                        motion_transition_counts[motion_id].item()
                    ),
                    "motion_label": dataset.motion_labels[motion_id],
                    "source_path": dataset.motion_source_paths[motion_id],
                }
            )
        initial_counts = torch.bincount(
            initial_motion_ids, minlength=motion_count
        )
        initial_distribution = {
            str(motion_id): int(count)
            for motion_id, count in enumerate(initial_counts.tolist())
            if count
        }
        recorder = base_env.nonfinite_diagnostics
        nonfinite = (
            {
                "total_events": 0,
                "policy_transitions": total_environment_transitions,
                "event_rate_per_policy_transition": 0.0,
            }
            if recorder is None
            else recorder.flush_summary(status="PASS")
        )
        episode = all_episodes.summary()
        root = all_root_errors.summary()
        result.update(
            {
                "status": "PASS",
                "configuration": {
                    "seed": int(args.seed),
                    "num_envs": int(args.num_envs),
                    "rollout_iterations": int(args.rollout_iterations),
                    "rollout_horizon": int(args.rollout_horizon),
                    "total_environment_transitions": total_environment_transitions,
                    "dataset_mem_path": str(args.dataset),
                    "dataset_device": str(dataset.ds_device),
                    "teacher_dataset_device": str(teacher_dataset.ds_device),
                    "deterministic_mean_action": True,
                    "vecnorm_mode": "eval",
                    "schedule_progress": schedule_progress,
                    "headless": True,
                    "wandb": "disabled",
                    "step_dt": float(base_env.step_dt),
                    "physics_dt": float(base_env.physics_dt),
                    "decimation": int(base_env.decimation),
                },
                "contracts": {
                    "policy_observation_shape": list(carry["policy"].shape),
                    "teacher_latent_dim": int(cfg.algo.latent_dim),
                    "action_dim": int(base_env.action_dim),
                    "action_equals_mean_max_abs_delta": action_mean_max_abs_delta,
                    "qpos_finite": True,
                    "qvel_finite": True,
                    "torque_finite": True,
                    "joint_target_finite": True,
                    "ctrl_finite": True,
                },
                "episode": episode,
                "root_tracking_error": root,
                "motion_coverage": {
                    "dataset_motion_count": motion_count,
                    "sampled_unique_motion_count": int(nonzero_motion_ids.numel()),
                    "sampled_motion_fraction": (
                        float(nonzero_motion_ids.numel()) / motion_count
                        if motion_count
                        else 0.0
                    ),
                    "initial_environment_distribution": initial_distribution,
                    "sampled_motion_id_distribution": motion_distribution,
                },
                "nonfinite": nonfinite,
                "purity": {
                    "optimizer_step_attempts": dict(optimizer_step_attempts),
                    "policy_parameter_delta": parameter_delta,
                    "vecnorm_state_delta": vecnorm_delta,
                    "no_optimizer_step": True,
                    "policy_parameters_unchanged": True,
                    "vecnorm_unchanged": True,
                },
                "throughput": {
                    "elapsed_s": elapsed_total,
                    "environment_transitions_per_second": (
                        total_environment_transitions / elapsed_total
                    ),
                    "policy_steps_per_second": (
                        int(args.rollout_iterations)
                        * int(args.rollout_horizon)
                        / elapsed_total
                    ),
                },
                "memory": {
                    "gpu_peak_allocated_mib": int(
                        torch.cuda.max_memory_allocated(device) // MIB
                    ),
                    "gpu_peak_reserved_mib": int(
                        torch.cuda.max_memory_reserved(device) // MIB
                    ),
                    "gpu_sampled_device_used_peak_mib": max(
                        row["gpu_device_used_mib"] for row in memory_samples
                    ),
                    "cpu_peak_rss_mib": max(
                        row["cpu_peak_rss_mib"] for row in memory_samples
                    ),
                    "samples": memory_samples,
                },
                "rollouts": rollout_rows,
            }
        )
        _write_json(output_dir / "evaluation.json", result)
        print(
            "TEACHER_EVAL_RESULT=PASS "
            f"episodes={episode['completed_episodes']} "
            f"reward={episode['mean_episode_reward']} "
            f"unique_motions={nonzero_motion_ids.numel()} "
            f"nonfinite={nonfinite.get('total_events', 0)}",
            flush=True,
        )
        return result
    except BaseException as error:
        result.update(
            {
                "status": "ABORTED",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        if env is not None:
            recorder = getattr(env.base_env, "nonfinite_diagnostics", None)
            if recorder is not None:
                result["nonfinite"] = recorder.flush_summary(status="ABORTED")
        _write_json(output_dir / "evaluation.json", result)
        raise
    finally:
        if env is not None:
            env.close(raise_if_closed=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="roban/main/shard_000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-iterations", type=int, default=100)
    parser.add_argument("--rollout-horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    run_evaluation(args)


if __name__ == "__main__":
    main()
