"""Diagnostics for bounded, real PPO smoke runs through ``scripts/train.py``.

The helpers in this module are deliberately opt-in.  They observe the regular
rollout and ``PPOPolicy.train_op`` path, save the repository's normal checkpoint
payload, and reload it through ``make_env_policy``.  They do not replace any
loss, backward pass, optimiser step, or phase-switch behaviour.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import resource
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.envs.utils import ExplorationType, set_exploration_type

from active_adaptation.learning.ppo.ppo import student_estimator_mse
import active_adaptation as aa


MIB = 1024 * 1024
EXPECTED_OBSERVATION_DIM = 1313
EXPECTED_LATENT_DIM = 40
EXPECTED_ACTOR_INPUT_DIM = 1353
EXPECTED_ACTION_DIM = 21
EXPECTED_DATASET_MEM_PATH = "_smoke/roban/main_subset"


def validate_smoke_schedule(
    *,
    num_envs: int,
    rollout_horizon: int,
    num_minibatches: int,
    total_frames: int,
    world_size: int = 1,
) -> dict[str, int]:
    """Validate the no-drop, single-process batching used by the smoke run."""
    values = {
        "num_envs": int(num_envs),
        "rollout_horizon": int(rollout_horizon),
        "num_minibatches": int(num_minibatches),
        "total_frames": int(total_frames),
        "world_size": int(world_size),
    }
    if values["world_size"] <= 0:
        raise ValueError("PPO smoke world_size must be positive")
    if not 1 <= values["num_envs"] <= 8192:
        raise ValueError("PPO smoke num_envs must be in [1, 8192] per rank")
    if values["rollout_horizon"] <= 1:
        raise ValueError("PPO smoke rollout_horizon must be greater than one")
    if values["num_minibatches"] <= 0:
        raise ValueError("PPO smoke num_minibatches must be positive")

    batch_size = values["num_envs"] * values["rollout_horizon"]
    if batch_size % values["num_minibatches"]:
        raise ValueError("rollout batch must divide evenly across PPO minibatches")
    # Estimator batching keeps complete time sequences and divides the env axis.
    if values["num_envs"] % values["num_minibatches"]:
        raise ValueError("num_envs must divide evenly across estimator minibatches")
    global_batch_size = batch_size * values["world_size"]
    if values["total_frames"] <= 0 or values["total_frames"] % global_batch_size:
        raise ValueError("total_frames must be a positive multiple of the global rollout batch")

    rollout_updates = values["total_frames"] // global_batch_size
    if not 2 <= rollout_updates <= 5:
        raise ValueError("PPO smoke must execute between 2 and 5 rollout updates")
    return {
        "batch_size": batch_size,
        "global_batch_size": global_batch_size,
        "minibatch_size": batch_size // values["num_minibatches"],
        "estimator_sequences_per_minibatch": (
            values["num_envs"] // values["num_minibatches"]
        ),
        "rollout_updates": rollout_updates,
    }


def validate_pilot_schedule(
    *,
    num_envs: int,
    rollout_horizon: int,
    num_minibatches: int,
    total_frames: int,
    target_rollout_updates: int,
    world_size: int = 1,
) -> dict[str, int]:
    """Validate the exact, no-drop single-GPU batching for a formal PPO pilot."""
    values = {
        "num_envs": int(num_envs),
        "rollout_horizon": int(rollout_horizon),
        "num_minibatches": int(num_minibatches),
        "total_frames": int(total_frames),
        "target_rollout_updates": int(target_rollout_updates),
        "world_size": int(world_size),
    }
    if values["world_size"] <= 0:
        raise ValueError("PPO pilot world_size must be positive")
    if not 1 <= values["num_envs"] <= 8192:
        raise ValueError("PPO pilot num_envs must be in [1, 8192] per rank")
    if values["rollout_horizon"] <= 1:
        raise ValueError("PPO pilot rollout_horizon must be greater than one")
    if values["num_minibatches"] <= 0:
        raise ValueError("PPO pilot num_minibatches must be positive")

    batch_size = values["num_envs"] * values["rollout_horizon"]
    if batch_size % values["num_minibatches"]:
        raise ValueError("rollout batch must divide evenly across PPO minibatches")
    if values["num_envs"] % values["num_minibatches"]:
        raise ValueError("num_envs must divide evenly across estimator minibatches")
    global_batch_size = batch_size * values["world_size"]
    if values["total_frames"] <= 0 or values["total_frames"] % global_batch_size:
        raise ValueError("total_frames must be a positive multiple of the global rollout batch")

    rollout_updates = values["total_frames"] // global_batch_size
    if rollout_updates != values["target_rollout_updates"]:
        raise ValueError(
            f"PPO pilot requires {values['target_rollout_updates']} rollout updates, "
            f"got {rollout_updates}"
        )
    return {
        "batch_size": batch_size,
        "global_batch_size": global_batch_size,
        "minibatch_size": batch_size // values["num_minibatches"],
        "estimator_sequences_per_minibatch": (
            values["num_envs"] // values["num_minibatches"]
        ),
        "rollout_updates": rollout_updates,
    }


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    # TorchRL containers (for example ProbabilisticActor) also expose a regular
    # ``module`` child.  Only DDP adds the wrapper prefix stripped by PPOPolicy's
    # checkpoint implementation.
    return module.module if isinstance(module, DDP) else module


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _unwrap(module).named_parameters()
    }


def parameter_delta(
    before: Mapping[str, torch.Tensor], module: torch.nn.Module
) -> dict[str, Any]:
    after = dict(_unwrap(module).named_parameters())
    missing = sorted(set(before) - set(after))
    unexpected = sorted(set(after) - set(before))
    if missing or unexpected:
        raise AssertionError(
            f"parameter keys changed: missing={missing}, unexpected={unexpected}"
        )
    changed = []
    max_abs_delta = 0.0
    for name, old in before.items():
        new = after[name].detach().cpu()
        delta = float((new - old).abs().max().item()) if new.numel() else 0.0
        max_abs_delta = max(max_abs_delta, delta)
        if delta > 0.0:
            changed.append(name)
    return {
        "changed": bool(changed),
        "changed_parameter_count": len(changed),
        "changed_parameters": changed,
        "max_abs_delta": max_abs_delta,
    }


def gradient_report(module: torch.nn.Module) -> dict[str, Any]:
    missing = []
    nonfinite = []
    nonzero_tensors = 0
    squared_norm = 0.0
    parameter_count = 0
    for name, parameter in _unwrap(module).named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_count += 1
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(grad).all().item()):
            nonfinite.append(name)
            continue
        if bool(torch.count_nonzero(grad).item()):
            nonzero_tensors += 1
        squared_norm += float(grad.detach().double().square().sum().item())
    return {
        "parameter_count": parameter_count,
        "missing_gradients": missing,
        "nonfinite_gradients": nonfinite,
        "nonzero_gradient_tensors": nonzero_tensors,
        "gradient_norm": math.sqrt(squared_norm),
        "finite": not nonfinite,
    }


def optimizer_finite_report(optimizer) -> dict[str, Any]:
    """Summarize every tensor held by an optimizer without mutating it."""
    tensor_count = 0
    element_count = 0
    nonfinite = []
    for parameter_index, state in enumerate(optimizer.state.values()):
        for state_name, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            tensor_count += 1
            element_count += int(value.numel())
            if not bool(torch.isfinite(value).all().item()):
                nonfinite.append(f"{parameter_index}:{state_name}")
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonfinite_tensors": nonfinite,
        "finite": not nonfinite,
    }


def state_key_report(
    saved_state: Mapping[str, Any], module: torch.nn.Module
) -> dict[str, list[str]]:
    current_keys = set(_unwrap(module).state_dict())
    saved_keys = set(saved_state)
    return {
        "missing_keys": sorted(current_keys - saved_keys),
        "unexpected_keys": sorted(saved_keys - current_keys),
    }


def _module_state_max_abs_delta(
    saved_state: Mapping[str, torch.Tensor], module: torch.nn.Module
) -> float:
    current = _unwrap(module).state_dict()
    report = state_key_report(saved_state, module)
    if report["missing_keys"] or report["unexpected_keys"]:
        return math.inf
    maximum = 0.0
    for name, old in saved_state.items():
        new = current[name].detach().cpu()
        old_cpu = old.detach().cpu()
        if new.numel():
            maximum = max(maximum, float((new - old_cpu).abs().max().item()))
    return maximum


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


def _assert_finite(label: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        bad = int((~torch.isfinite(value)).sum().item())
        raise FloatingPointError(f"{label} contains {bad} NaN/Inf value(s)")


def _nonfinite_summary(value: torch.Tensor, num_envs: int) -> dict[str, Any] | None:
    finite = torch.isfinite(value)
    if bool(finite.all().item()):
        return None
    bad = ~finite
    summary: dict[str, Any] = {
        "shape": list(value.shape),
        "nonfinite_count": int(bad.sum().item()),
    }
    if value.ndim and value.shape[0] == num_envs:
        bad_by_env = bad.reshape(num_envs, -1).any(dim=1)
        summary["environment_indices"] = (
            bad_by_env.nonzero().reshape(-1).detach().cpu().tolist()
        )
    finite_values = value[finite]
    if finite_values.numel():
        summary["finite_min"] = float(finite_values.min().item())
        summary["finite_max"] = float(finite_values.max().item())
    return summary


def root_tracking_error_summary(
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_quat: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return actual root XY/Z/geodesic errors for diagnostic logging."""
    delta = current_pos - target_pos
    xy = torch.linalg.vector_norm(delta[..., :2], dim=-1)
    z = delta[..., 2].abs()
    current_quat = current_quat / torch.linalg.vector_norm(
        current_quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    target_quat = target_quat / torch.linalg.vector_norm(
        target_quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    dot = (current_quat * target_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    orientation = 2.0 * torch.acos(dot)
    if valid is not None:
        valid = valid.reshape(-1)
        xy = xy.reshape(-1)[valid]
        z = z.reshape(-1)[valid]
        orientation = orientation.reshape(-1)[valid]

    def summarize(value: torch.Tensor) -> dict[str, float | int | None]:
        value = value.reshape(-1)
        if not value.numel():
            return {"count": 0, "mean": None, "p95": None, "max": None}
        _assert_finite("root_tracking_error", value)
        value = value.double()
        return {
            "count": int(value.numel()),
            "mean": float(value.mean().item()),
            "p95": float(torch.quantile(value, 0.95).item()),
            "max": float(value.max().item()),
        }

    return {
        "root_xy_position_error_m": summarize(xy),
        "root_z_error_m": summarize(z),
        "root_orientation_geodesic_error_rad": summarize(orientation),
    }


class PPOSmokeDiagnostics:
    """Observe and attest a bounded PPO smoke or pilot without changing its algorithm."""

    _MODULE_NAMES = (
        "actor_teacher",
        "actor_student",
        "encoder_priv",
        "adapt_module",
        "critic",
    )
    _EXPECTED_CHANGED = {
        "train": ("actor_teacher", "encoder_priv", "critic", "adapt_module"),
        "adapt": ("adapt_module",),
        "finetune": ("actor_student", "critic"),
    }
    _EXPECTED_FROZEN = {
        "train": ("actor_student",),
        "adapt": ("actor_teacher", "actor_student", "encoder_priv", "critic"),
        "finetune": ("actor_teacher", "encoder_priv", "adapt_module"),
    }
    _EXPECTED_PREVIOUS = {"train": None, "adapt": "train", "finetune": "adapt"}

    def __init__(
        self,
        cfg: DictConfig,
        env,
        policy,
        vecnorm,
        *,
        total_iters: int,
        audit_kind: str = "smoke",
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.vecnorm = vecnorm
        self.base_env = env.base_env
        self.audit_kind = str(audit_kind)
        if self.audit_kind not in {"smoke", "pilot"}:
            raise ValueError(f"unsupported PPO audit kind: {self.audit_kind}")
        self.audit_label = f"PPO_{self.audit_kind.upper()}"
        self.total_iters = int(total_iters)
        self.phase = str(cfg.algo.phase)
        if self.phase not in self._EXPECTED_CHANGED:
            raise ValueError(f"unsupported {self.audit_label} phase: {self.phase}")
        if self.audit_kind == "pilot" and self.phase != "train":
            raise ValueError("the formal Roban pilot only supports the Teacher train phase")

        self.world_size = aa.get_world_size()
        self.rank = aa.get_local_rank()
        allow_ddp = bool(cfg.get(self.audit_kind, {}).get("allow_ddp", False))
        if self.world_size != 1 and not allow_ddp:
            raise RuntimeError(f"{self.audit_label} DDP requires allow_ddp=true")
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        expected_visible = "3" if self.world_size == 1 else "3,4,7"
        if visible_devices != expected_visible:
            raise RuntimeError(
                f"{self.audit_label} requires CUDA_VISIBLE_DEVICES={expected_visible}, "
                f"got {visible_devices!r}"
            )
        if str(self.base_env.device) != f"cuda:{self.rank}":
            raise RuntimeError(
                f"{self.audit_label} expects logical cuda:{self.rank}, got {self.base_env.device}"
            )

        audit_cfg = cfg.get(self.audit_kind, {})
        self.audit_cfg = audit_cfg
        output_dir = Path(str(audit_cfg.get("output_dir", ""))).resolve()
        if self.audit_kind == "smoke":
            smoke_root_raw = os.environ.get("PPO_SMOKE_ROOT")
            if not smoke_root_raw:
                raise RuntimeError("PPO_SMOKE_ROOT must be set for PPO smoke")
            smoke_root = Path(smoke_root_raw).resolve()
            if output_dir.parent != smoke_root or output_dir.name != self.phase:
                raise RuntimeError(
                    f"smoke output_dir must be {smoke_root / self.phase}, got {output_dir}"
                )
        else:
            pilot_root_raw = os.environ.get("PILOT_ROOT")
            if not pilot_root_raw:
                raise RuntimeError("PILOT_ROOT must be set for PPO pilot")
            pilot_root = Path(pilot_root_raw).resolve()
            if output_dir != pilot_root:
                raise RuntimeError(
                    f"pilot output_dir must be {pilot_root}, got {output_dir}"
                )
        self.root_output_dir = output_dir
        if self.world_size > 1 and self.rank != 0:
            output_dir = output_dir / f"rank_{self.rank}"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_offset = int(audit_cfg.get("rollout_offset", 0))
        self.global_policy_frame_offset = int(
            audit_cfg.get("global_policy_frame_offset", 0)
        )
        self.checkpoint_interval = int(audit_cfg.get("checkpoint_interval", -1))
        if self.rollout_offset < 0 or self.global_policy_frame_offset < 0:
            raise ValueError("pilot rollout/frame offsets must be non-negative")
        if self.global_policy_frame_offset % int(self.base_env.num_envs):
            raise ValueError(
                "global_policy_frame_offset must divide by num_envs exactly"
            )

        schedule_kwargs = {
            "num_envs": self.base_env.num_envs,
            "rollout_horizon": cfg.algo.train_every,
            "num_minibatches": cfg.algo.num_minibatches,
            "total_frames": cfg.total_frames,
            "world_size": self.world_size,
        }
        if self.audit_kind == "smoke":
            schedule = validate_smoke_schedule(**schedule_kwargs)
        else:
            target_rollouts = int(audit_cfg.get("target_rollout_iterations", 20))
            schedule = validate_pilot_schedule(
                **schedule_kwargs,
                target_rollout_updates=target_rollouts,
            )
            expected_num_envs = int(audit_cfg.get("expected_num_envs", 256))
            if int(self.base_env.num_envs) != expected_num_envs:
                raise ValueError(
                    f"PPO pilot expected {expected_num_envs} envs, "
                    f"got {self.base_env.num_envs}"
                )
        if schedule["rollout_updates"] != int(total_iters):
            raise AssertionError(
                f"derived rollout updates {schedule['rollout_updates']} != train.py {total_iters}"
            )
        if self.audit_kind == "smoke" and int(cfg.algo.ppo_epochs) != 1:
            raise ValueError("PPO smoke expects exactly one PPO learning epoch")
        if bool(cfg.eval_render) or bool(cfg.app.enable_cameras):
            raise RuntimeError(f"{self.audit_label} forbids rendering and cameras")
        if not bool(cfg.headless) or not bool(cfg.app.headless):
            raise RuntimeError(f"{self.audit_label} must remain headless")
        if str(cfg.wandb.mode) != "disabled":
            raise RuntimeError(f"{self.audit_label} must keep W&B disabled")
        if bool(cfg.train_record.enabled):
            raise RuntimeError(f"{self.audit_label} must keep train_record disabled")

        dataset = self.base_env.command_manager.dataset
        teacher_dataset = dataset.teacher_dataset
        mem_root = Path(os.environ["MEMPATH"]).resolve()
        loaded_paths = []
        for raw_path in teacher_dataset.dataset_paths:
            path = Path(raw_path)
            loaded_paths.append((path if path.is_absolute() else mem_root / path).resolve())
        expected_dataset_mem_path = str(
            audit_cfg.get("dataset_mem_path", EXPECTED_DATASET_MEM_PATH)
        )
        expected_dataset = (mem_root / expected_dataset_mem_path).resolve()
        if loaded_paths != [expected_dataset]:
            raise AssertionError(
                f"{self.audit_label} loaded datasets {loaded_paths}, expected {[expected_dataset]}"
            )
        if str(dataset.ds_device) != "cpu" or str(teacher_dataset.ds_device) != "cpu":
            raise AssertionError(f"{self.audit_label} dataset and cache must stay on CPU")
        if int(self.base_env.action_dim) != EXPECTED_ACTION_DIM:
            raise AssertionError(
                f"action dim {self.base_env.action_dim} != {EXPECTED_ACTION_DIM}"
            )
        if int(cfg.algo.latent_dim) != EXPECTED_LATENT_DIM:
            raise AssertionError(
                f"latent dim {cfg.algo.latent_dim} != {EXPECTED_LATENT_DIM}"
            )
        if not bool(policy.student_latent_quantizer_enabled):
            raise AssertionError(f"Roban {self.audit_label} requires the Student quantizer")
        if list(policy.student_latent_quantizer.parameters()):
            raise AssertionError("Student quantizer unexpectedly has trainable parameters")

        self.nonfinite_recorder = getattr(
            self.base_env, "nonfinite_diagnostics", None
        )
        requested_nonfinite_diagnostics = bool(
            audit_cfg.get("nonfinite_diagnostics", {}).get("enabled", False)
        )
        if requested_nonfinite_diagnostics and self.nonfinite_recorder is None:
            raise AssertionError(
                "pilot requested nonfinite diagnostics but the environment recorder is absent"
            )
        if self.nonfinite_recorder is not None:
            if self.audit_kind != "pilot":
                raise AssertionError("event-level nonfinite diagnostics are pilot-only")
            if self.nonfinite_recorder.output_dir != self.output_dir:
                raise AssertionError(
                    "nonfinite recorder and pilot output directories differ"
                )

        OmegaConf.save(cfg, self.output_dir / "resolved_config.yaml")
        self.optimizer_step_counts = {
            "teacher": 0,
            "student": 0,
            "critic": 0,
            "estimator": 0,
        }
        self._install_optimizer_step_counters()
        self._initial_parameters = {
            name: _parameter_snapshot(getattr(policy, name))
            for name in self._MODULE_NAMES
        }
        self._pre_update_parameters: dict[str, dict[str, torch.Tensor]] = {}
        self._rollout_reward_min = math.inf
        self._rollout_reward_max = -math.inf
        self._motion_count = len(dataset.motion_labels)
        self._sampled_motion_ids = torch.zeros(
            self._motion_count, dtype=torch.bool, device=self.base_env.device
        )
        self._reset_iteration_statistics()
        self._last_fixed_input = None
        self._previous_transition_context: dict[str, torch.Tensor] | None = None
        self._last_metrics: dict[str, Any] = {}
        self._gradient_reports: dict[str, Any] = {}
        self._phase_load_report = self._verify_input_checkpoint()
        if self._phase_load_report is not None:
            self._write_json("checkpoint_load_report.json", self._phase_load_report)

        self.metrics: dict[str, Any] = {
            "status": "RUNNING",
            "audit_kind": self.audit_kind,
            "phase": self.phase,
            "python_pid": os.getpid(),
            "cuda_visible_devices": visible_devices,
            "logical_device": str(self.base_env.device),
            "configuration": {
                **schedule,
                "num_envs": int(self.base_env.num_envs),
                "rollout_horizon": int(cfg.algo.train_every),
                "num_minibatches": int(cfg.algo.num_minibatches),
                "ppo_learning_epochs": int(cfg.algo.ppo_epochs),
                "estimator_learning_epochs": 2,
                "total_frames": int(cfg.total_frames),
                "total_frames_semantics": "global_budget_across_all_ranks",
                "world_size": self.world_size,
                "rank": self.rank,
                "per_rank_num_envs": int(self.base_env.num_envs),
                "global_num_envs": int(self.base_env.num_envs) * self.world_size,
                "rollout_offset": self.rollout_offset,
                "global_policy_frame_offset": self.global_policy_frame_offset,
                "checkpoint_interval": self.checkpoint_interval,
                "optimizer": str(cfg.algo.optimizer),
                "dataset_path": str(expected_dataset),
                "dataset_mem_path": expected_dataset_mem_path,
                "dataset_device": str(teacher_dataset.ds_device),
                "step_dt": float(self.base_env.step_dt),
                "physics_dt": float(self.base_env.physics_dt),
                "decimation": int(self.base_env.decimation),
            },
            "rollouts": [],
            "optimizer_step_counts": self.optimizer_step_counts,
        }
        self._sample_memory("after_initialization")
        self._flush_metrics()
        print(
            f"{self.audit_label}_START "
            f"phase={self.phase} pid={os.getpid()} schedule={schedule}",
            flush=True,
        )

    def _install_optimizer_step_counters(self) -> None:
        optimizers = {
            "teacher": self.policy.opt_teacher,
            "student": self.policy.opt_student,
            "critic": self.policy.opt_critic,
            "estimator": self.policy.opt_estimator,
        }
        for name, optimizer in optimizers.items():
            original_step = optimizer.step

            def counted_step(*args, _name=name, _step=original_step, **kwargs):
                self.optimizer_step_counts[_name] += 1
                return _step(*args, **kwargs)

            optimizer.step = counted_step

    def _verify_input_checkpoint(self) -> dict[str, Any] | None:
        expected_previous = self._EXPECTED_PREVIOUS[self.phase]
        raw_path = self.cfg.checkpoint_path
        if expected_previous is None:
            if raw_path is None:
                return None
            if not (
                self.audit_kind == "pilot"
                and bool(self.audit_cfg.get("allow_teacher_resume", False))
            ):
                raise AssertionError("train smoke must start without a checkpoint")
            expected_previous = "train"
        if raw_path is None or str(raw_path).startswith("run:"):
            raise AssertionError(
                f"{self.phase} smoke must use the prior local checkpoint path"
            )
        checkpoint_path = Path(str(raw_path)).resolve()
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise FileNotFoundError(f"input checkpoint is missing or empty: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy_state = checkpoint.get("policy", {})
        source_phase = policy_state.get("last_phase")
        if source_phase != expected_previous:
            raise AssertionError(
                f"{self.phase} expected {expected_previous!r} checkpoint, got {source_phase!r}"
            )

        key_reports = {}
        max_abs_deltas = {}
        for name, module in self._modules().items():
            saved = policy_state.get(name, {})
            key_reports[name] = state_key_report(saved, module)
            comparison_state = saved
            if self.phase == "adapt" and name == "actor_student":
                # Official phase switch initializes Student from Teacher.
                comparison_state = policy_state["actor_teacher"]
            elif self.phase == "train" and name == "actor_student":
                # Loading a Teacher checkpoint intentionally refreshes the
                # unused Student actor from the Teacher actor.
                comparison_state = policy_state["actor_teacher"]
            max_abs_deltas[name] = _module_state_max_abs_delta(comparison_state, module)
        critical_missing = {
            name: report
            for name, report in key_reports.items()
            if report["missing_keys"] or report["unexpected_keys"]
        }
        if critical_missing:
            raise AssertionError(f"checkpoint key mismatch: {critical_missing}")
        if any(not math.isfinite(delta) or delta != 0.0 for delta in max_abs_deltas.values()):
            raise AssertionError(f"checkpoint did not load exactly: {max_abs_deltas}")
        return {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "expected_previous_phase": expected_previous,
            "source_phase": source_phase,
            "teacher_continuation": self.phase == "train",
            "official_teacher_to_student_hard_copy": self.phase == "adapt",
            "module_key_reports": key_reports,
            "loaded_module_max_abs_delta": max_abs_deltas,
            "critical_missing_or_unexpected": {},
            "pass": True,
        }

    def _modules(self) -> dict[str, torch.nn.Module]:
        modules = {name: getattr(self.policy, name) for name in self._MODULE_NAMES}
        modules["student_latent_quantizer"] = self.policy.student_latent_quantizer
        return modules

    def _write_json(self, file_name: str, payload: Any) -> None:
        path = self.output_dir / file_name
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _flush_metrics(self) -> None:
        self._write_json("metrics.json", self.metrics)

    def _reset_iteration_statistics(self) -> None:
        device = self.base_env.device
        reward_group_count = len(self.policy.reward_groups)
        self._iteration_reward_sum = torch.zeros(
            reward_group_count, device=device, dtype=torch.float64
        )
        self._iteration_transition_count = 0
        self._iteration_episode_count = torch.zeros((), device=device, dtype=torch.long)
        self._iteration_terminated_count = torch.zeros((), device=device, dtype=torch.long)
        self._iteration_truncated_count = torch.zeros((), device=device, dtype=torch.long)
        self._iteration_immediate_termination_count = torch.zeros(
            (), device=device, dtype=torch.long
        )
        self._iteration_episode_stat_sums: dict[str, torch.Tensor] = {}
        self._iteration_root_current_pos: list[torch.Tensor] = []
        self._iteration_root_target_pos: list[torch.Tensor] = []
        self._iteration_root_current_quat: list[torch.Tensor] = []
        self._iteration_root_target_quat: list[torch.Tensor] = []
        self._iteration_motion_ids = torch.zeros(
            self._motion_count, dtype=torch.bool, device=device
        )

    def _iteration_episode_summary(self) -> dict[str, Any]:
        episode_count = int(self._iteration_episode_count.item())
        terminated_count = int(self._iteration_terminated_count.item())
        truncated_count = int(self._iteration_truncated_count.item())
        immediate_count = int(self._iteration_immediate_termination_count.item())
        episode_means = {}
        episode_sums = {
            key: float(value.item())
            for key, value in sorted(self._iteration_episode_stat_sums.items())
        }
        if episode_count:
            episode_means = {
                key: float((value / episode_count).item())
                for key, value in sorted(self._iteration_episode_stat_sums.items())
            }
        reward_returns = [
            value
            for key, value in episode_means.items()
            if key.endswith("/return")
        ]
        mean_episode_reward = sum(reward_returns) if reward_returns else None
        episode_length_sum = float(
            self._iteration_episode_stat_sums.get(
                "episode_len", torch.zeros((), device=self.base_env.device)
            ).item()
        )
        per_step_episode_stats = {}
        if episode_length_sum > 0.0:
            per_step_episode_stats = {
                key: value / episode_length_sum
                for key, value in episode_sums.items()
                if key != "episode_len"
            }

        transition_count = int(self._iteration_transition_count)
        step_reward_means = {}
        if transition_count:
            reward_values = (
                self._iteration_reward_sum / transition_count
            ).detach().cpu().tolist()
            step_reward_means = {
                str(name): float(value)
                for name, value in zip(self.policy.reward_groups, reward_values)
            }
        return {
            "completed_episodes": episode_count,
            "terminated_episodes": terminated_count,
            "truncated_episodes": truncated_count,
            "termination_rate_per_completed_episode": (
                terminated_count / episode_count if episode_count else None
            ),
            "termination_rate_per_transition": (
                terminated_count / transition_count if transition_count else None
            ),
            "immediate_terminations_le_10_steps": immediate_count,
            "immediate_termination_rate_per_completed_episode": (
                immediate_count / episode_count if episode_count else None
            ),
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": episode_means.get("episode_len"),
            "mean_episode_stats": episode_means,
            "episode_stat_sums": episode_sums,
            "episode_length_sum": episode_length_sum,
            "mean_episode_stats_per_step": per_step_episode_stats,
            "mean_step_reward_by_group": step_reward_means,
        }

    def _sample_memory(self, label: str) -> dict[str, int | str]:
        device = str(self.base_env.device)
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        cpu_peak_rss_mib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        cpu_current_rss_mib = None
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    cpu_current_rss_mib = float(line.split()[1]) / 1024.0
                    break
        except OSError:
            pass
        sample = {
            "label": label,
            "used_mib": int((total_bytes - free_bytes) // MIB),
            "free_mib": int(free_bytes // MIB),
            "torch_allocated_mib": int(torch.cuda.memory_allocated(device) // MIB),
            "torch_reserved_mib": int(torch.cuda.memory_reserved(device) // MIB),
            "cpu_peak_rss_mib": cpu_peak_rss_mib,
            "cpu_current_rss_mib": cpu_current_rss_mib,
        }
        self.metrics.setdefault("gpu_memory_samples", []).append(sample)
        return sample

    def on_reset(self, carry) -> None:
        observation = carry["policy"]
        expected = (int(self.base_env.num_envs), EXPECTED_OBSERVATION_DIM)
        if tuple(observation.shape) != expected:
            raise AssertionError(
                f"reset observation shape {tuple(observation.shape)} != {expected}"
            )
        _assert_finite("reset.policy", observation)
        _assert_finite("reset.qpos", self.base_env.sim.data.qpos)
        _assert_finite("reset.qvel", self.base_env.sim.data.qvel)
        self.metrics["reset_observation_shape"] = list(observation.shape)
        self.metrics["reset_finite"] = True
        self._flush_metrics()

    def before_transition(self, *, iteration: int, rollout_step: int) -> None:
        """Provide exact rollout/frame counters to the simulator event probe."""
        if self.nonfinite_recorder is None:
            return
        local_policy_step = int(iteration) * int(self.cfg.algo.train_every) + int(
            rollout_step
        )
        global_policy_frame = self.global_policy_frame_offset + (
            local_policy_step + 1
        ) * int(self.base_env.num_envs) * self.world_size
        global_policy_step = (
            self.global_policy_frame_offset // int(self.base_env.num_envs)
        ) + local_policy_step
        self.base_env.set_nonfinite_diagnostic_context(
            policy_iteration=self.rollout_offset + int(iteration) + 1,
            global_policy_frame=global_policy_frame,
            global_physics_tick_base=(
                global_policy_step * int(self.base_env.decimation)
            ),
        )

    def observe_transition(self, td, *, iteration: int, rollout_step: int) -> None:
        observation = td["policy"]
        action = td["action"]
        reward = td["next", "reward"]
        done = td["next", "done"].reshape(-1)
        terminated = td["next", "terminated"].reshape(-1)
        truncated = td["next", "truncated"].reshape(-1)
        value = td["state_value"]
        next_value = td["next", "state_value"]
        stats = td["next", "stats"]
        if tuple(observation.shape) != (
            int(self.base_env.num_envs),
            EXPECTED_OBSERVATION_DIM,
        ):
            raise AssertionError(f"policy observation shape mismatch: {observation.shape}")
        if tuple(action.shape) != (int(self.base_env.num_envs), EXPECTED_ACTION_DIM):
            raise AssertionError(f"action shape mismatch: {action.shape}")
        validation_tensors = (
            ("policy_observation", observation),
            ("next_policy_observation", td["next", "policy"]),
            ("next_private_observation", td["next", "priv"]),
            ("next_critic_private_observation", td["next", "priv_critic"]),
            ("action", action),
            ("reward", reward),
            ("value", value),
            ("next_value", next_value),
            ("qpos", self.base_env.sim.data.qpos),
            ("qvel", self.base_env.sim.data.qvel),
            ("torque", self.base_env.scene["robot"].data.qfrc_actuator),
        )
        failures = {
            label: summary
            for label, tensor in validation_tensors
            if (summary := _nonfinite_summary(tensor, int(self.base_env.num_envs)))
            is not None
        }
        if failures:
            recovered_ids = (
                self.base_env._nonfinite_state_mask.nonzero()
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()
            )
            bad_environment_indices = sorted(
                recovered_ids
                or {
                    int(env_id)
                    for summary in failures.values()
                    for env_id in summary.get("environment_indices", [])
                }
            )
            bad_env_ids = torch.tensor(
                bad_environment_indices,
                device=self.base_env.device,
                dtype=torch.long,
            )

            def selected(tensor):
                if not isinstance(tensor, torch.Tensor):
                    return None
                return tensor[bad_env_ids].detach().cpu()

            component_failures = {}
            for key, tensor in stats.items(True, True):
                if not isinstance(tensor, torch.Tensor):
                    continue
                label = "/".join(key) if isinstance(key, tuple) else str(key)
                summary = _nonfinite_summary(tensor, int(self.base_env.num_envs))
                if summary is not None:
                    component_failures[label] = summary

            derived_failures = {}
            asset_data = self.base_env.scene["robot"].data
            for name in (
                "joint_pos",
                "joint_vel",
                "joint_acc",
                "actuator_force",
                "qfrc_actuator",
                "body_link_pos_w",
                "body_link_quat_w",
                "body_link_lin_vel_w",
                "body_link_ang_vel_w",
            ):
                tensor = getattr(asset_data, name, None)
                if not isinstance(tensor, torch.Tensor):
                    continue
                summary = _nonfinite_summary(tensor, int(self.base_env.num_envs))
                if summary is not None:
                    derived_failures[name] = summary

            command = self.base_env.command_manager
            action_manager = self.base_env.action_manager
            dataset = command.dataset
            motion_ids = selected(dataset._motion_ids_A).tolist()
            motion_context = []
            for env_id, motion_id in zip(bad_environment_indices, motion_ids):
                motion_context.append(
                    {
                        "environment_index": env_id,
                        "motion_id": int(motion_id),
                        "motion_label": dataset.motion_labels[int(motion_id)],
                        "motion_source_path": dataset.motion_source_paths[int(motion_id)],
                    }
                )

            current_context = {
                "action": selected(action),
                "action_loc": selected(td.get("loc", None)),
                "action_scale": selected(td.get("scale", None)),
                "applied_action": selected(action_manager.applied_action),
                "joint_position_target": selected(asset_data.joint_pos_target),
                "joint_offset": selected(action_manager.offset),
                "action_alpha": selected(action_manager.alpha),
                "action_delay": selected(action_manager.delay),
                "episode_length": selected(self.base_env.episode_length_buf),
                "motion_frame": selected(command.t),
                "motion_length": selected(command.lengths),
            }
            motor_randomization = self.base_env.randomizations.get(
                "motor_params_implicit"
            )
            if motor_randomization is not None:
                current_context["motor_stiffness_scale"] = selected(
                    motor_randomization._obs_kp_scale
                )
                current_context["motor_damping_scale"] = selected(
                    motor_randomization._obs_kd_scale
                )
                current_context["motor_armature_scale"] = selected(
                    motor_randomization._obs_arm_scale
                )
            previous_context = (
                None
                if self._previous_transition_context is None
                else {
                    name: selected(tensor)
                    for name, tensor in self._previous_transition_context.items()
                }
            )

            report = {
                "iteration": int(iteration),
                "rollout_step": int(rollout_step),
                "environment_indices": bad_environment_indices,
                "tensor_failures": failures,
                "reward_or_termination_stat_failures": component_failures,
                "derived_state_failures": derived_failures,
                "motion_context": motion_context,
                "previous_transition_context": previous_context,
                "current_transition_context": current_context,
            }
            self.metrics["status"] = "FAILED_NONFINITE"
            self.metrics["failure"] = report
            self._write_json("nonfinite_failure.json", report)
            self._flush_metrics()
            print(f"{self.audit_label}_NONFINITE {_jsonable(report)}", flush=True)
            labels = ", ".join(sorted(failures))
            raise FloatingPointError(f"non-finite tensors detected: {labels}")
        actor_input = td.get("_actor_inp", None)
        if actor_input is not None:
            if actor_input.shape[-1] != EXPECTED_ACTOR_INPUT_DIM:
                raise AssertionError(
                    f"actor input width {actor_input.shape[-1]} != {EXPECTED_ACTOR_INPUT_DIM}"
                )
            _assert_finite("actor_input", actor_input)
        self._previous_transition_context = {
            "qpos": self.base_env.sim.data.qpos.detach().clone(),
            "qvel": self.base_env.sim.data.qvel.detach().clone(),
            "action": action.detach().clone(),
            "applied_action": self.base_env.action_manager.applied_action.detach().clone(),
            "joint_position_target": self.base_env.scene[
                "robot"
            ].data.joint_pos_target.detach().clone(),
            "episode_length": self.base_env.episode_length_buf.detach().clone(),
            "motion_frame": self.base_env.command_manager.t.detach().clone(),
        }
        self._rollout_reward_min = min(
            self._rollout_reward_min, float(reward.min().item())
        )
        self._rollout_reward_max = max(
            self._rollout_reward_max, float(reward.max().item())
        )
        self._iteration_reward_sum.add_(reward.detach().double().sum(dim=0))
        self._iteration_transition_count += int(reward.shape[0])
        self._iteration_episode_count.add_(done.sum())
        self._iteration_terminated_count.add_(terminated.sum())
        self._iteration_truncated_count.add_(truncated.sum())
        episode_len = stats["episode_len"].reshape(-1)
        self._iteration_immediate_termination_count.add_(
            (terminated & (episode_len <= 10)).sum()
        )
        if bool(done.any().item()):
            for key, tensor in stats.items(True, True):
                if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != done.shape[0]:
                    continue
                flat_key = "/".join(key) if isinstance(key, tuple) else str(key)
                value = tensor[done].detach().double().sum()
                if flat_key not in self._iteration_episode_stat_sums:
                    self._iteration_episode_stat_sums[flat_key] = torch.zeros_like(value)
                self._iteration_episode_stat_sums[flat_key].add_(value)
        # step_and_maybe_reset has already reset completed rows.  Exclude those
        # rows so the actual state and the reward target refer to the same step.
        root_valid = ~done
        if bool(root_valid.any().item()):
            asset_data = self.base_env.scene["robot"].data
            command = self.base_env.command_manager
            self._iteration_root_current_pos.append(
                asset_data.root_link_pos_w[root_valid].detach().clone()
            )
            self._iteration_root_target_pos.append(
                command.reward_root_pos_w[root_valid].detach().clone()
            )
            self._iteration_root_current_quat.append(
                asset_data.root_link_quat_w[root_valid].detach().clone()
            )
            self._iteration_root_target_quat.append(
                command.reward_root_quat_w[root_valid].detach().clone()
            )
        sampled_ids = self.base_env.command_manager.dataset._motion_ids_A.long()
        self._iteration_motion_ids[sampled_ids] = True
        self._sampled_motion_ids[sampled_ids] = True
        self._last_fixed_input = td.select(
            "policy", "priv", "priv_critic", "is_init", strict=False
        )[: min(4, int(self.base_env.num_envs))].clone().cpu()
        self.metrics["last_transition"] = {
            "iteration": int(iteration),
            "rollout_step": int(rollout_step),
            "observation_shape": list(observation.shape),
            "action_shape": list(action.shape),
            "reward_shape": list(reward.shape),
            "actor_input_shape": list(actor_input.shape) if actor_input is not None else None,
            "qpos_finite": True,
            "qvel_finite": True,
            "torque_finite": True,
            "reward_finite": True,
            "value_finite": True,
        }

    def before_update(self, data, *, iteration: int) -> None:
        expected_batch_shape = (
            int(self.base_env.num_envs),
            int(self.cfg.algo.train_every),
        )
        if tuple(data.shape) != expected_batch_shape:
            raise AssertionError(
                f"rollout TensorDict shape {tuple(data.shape)} != {expected_batch_shape}"
            )
        self._pre_update_parameters = {
            name: _parameter_snapshot(getattr(self.policy, name))
            for name in self._EXPECTED_CHANGED[self.phase]
        }
        self._sample_memory(f"iteration_{iteration}_before_update")

    def after_update(
        self,
        data,
        train_metrics: Mapping[str, Any],
        *,
        iteration: int,
        rollout_time: float | None = None,
        training_time: float | None = None,
    ) -> None:
        for name, value in train_metrics.items():
            if isinstance(value, torch.Tensor):
                _assert_finite(f"metric.{name}", value)
            elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"metric.{name} is not finite: {value}")

        if self.phase in {"train", "finetune"}:
            for key in ("adv", "ret"):
                if key not in data.keys():
                    raise AssertionError(f"PPO update did not produce {key}")
                _assert_finite(key, data[key])

        changes = {}
        for name, snapshot in self._pre_update_parameters.items():
            changes[name] = parameter_delta(snapshot, getattr(self.policy, name))
            if not changes[name]["changed"]:
                raise AssertionError(
                    f"expected trained module {name} did not change in iteration {iteration}"
                )

        gradients = {}
        for name in self._EXPECTED_CHANGED[self.phase]:
            report = gradient_report(getattr(self.policy, name))
            gradients[name] = report
            if report["missing_gradients"] or report["nonfinite_gradients"]:
                raise AssertionError(f"invalid {name} gradients: {report}")
            if report["nonzero_gradient_tensors"] == 0 or report["gradient_norm"] <= 0.0:
                raise AssertionError(f"zero {name} gradients: {report}")
        self._gradient_reports[f"iteration_{iteration}"] = gradients

        optimizer_reports = {}
        for name, optimizer in {
            "teacher": self.policy.opt_teacher,
            "student": self.policy.opt_student,
            "critic": self.policy.opt_critic,
            "estimator": self.policy.opt_estimator,
        }.items():
            report = optimizer_finite_report(optimizer)
            optimizer_reports[name] = report
            if not report["finite"]:
                raise FloatingPointError(f"non-finite {name} optimizer state: {report}")

        timing = {}
        if rollout_time is not None and training_time is not None:
            rollout_time = float(rollout_time)
            training_time = float(training_time)
            if not math.isfinite(rollout_time) or rollout_time <= 0.0:
                raise FloatingPointError(f"invalid rollout_time: {rollout_time}")
            if not math.isfinite(training_time) or training_time <= 0.0:
                raise FloatingPointError(f"invalid training_time: {training_time}")
            frames = int(data.numel())
            timing = {
                "rollout_time_s": rollout_time,
                "training_time_s": training_time,
                "rollout_steps_per_second": frames / rollout_time,
                "end_to_end_steps_per_second": frames / (rollout_time + training_time),
            }

        rollout = {
            "iteration": int(iteration),
            "rollout_index": self.rollout_offset + int(iteration) + 1,
            "rollout_complete": True,
            "batch_shape": list(data.shape),
            "batch_size": int(data.numel()),
            "reward_finite": True,
            "reward_min": self._rollout_reward_min,
            "reward_max": self._rollout_reward_max,
            "advantage_finite": self.phase in {"train", "finetune"},
            "return_finite": self.phase in {"train", "finetune"},
            "metrics": _jsonable(train_metrics),
            "gradients": gradients,
            "parameter_changes": changes,
            "optimizer_step_counts": dict(self.optimizer_step_counts),
            "optimizer_finite": optimizer_reports,
            "nonfinite_state_recoveries_total": int(
                self.base_env.nonfinite_state_recovery_count
            ),
            "nonfinite_events_in_iteration": (
                0
                if self.nonfinite_recorder is None
                else int(
                    self.nonfinite_recorder.events_by_iteration.get(
                        str(self.rollout_offset + int(iteration) + 1), 0
                    )
                )
            ),
            "episode": self._iteration_episode_summary(),
            "root_tracking_error": (
                root_tracking_error_summary(
                    torch.cat(self._iteration_root_current_pos),
                    torch.cat(self._iteration_root_target_pos),
                    torch.cat(self._iteration_root_current_quat),
                    torch.cat(self._iteration_root_target_quat),
                )
                if self._iteration_root_current_pos
                else root_tracking_error_summary(
                    torch.empty((0, 3), device=self.base_env.device),
                    torch.empty((0, 3), device=self.base_env.device),
                    torch.empty((0, 4), device=self.base_env.device),
                    torch.empty((0, 4), device=self.base_env.device),
                )
            ),
            "motion_coverage": {
                "dataset_motion_count": self._motion_count,
                "sampled_unique_motions_in_rollout": int(
                    self._iteration_motion_ids.sum().item()
                ),
                "sampled_unique_motions_total": int(
                    self._sampled_motion_ids.sum().item()
                ),
            },
            "timing": timing,
            "gpu_memory": self._sample_memory(f"iteration_{iteration}_after_update"),
        }
        self.metrics["rollouts"].append(rollout)
        self.metrics["optimizer_step_counts"] = dict(self.optimizer_step_counts)
        self._rollout_reward_min = math.inf
        self._rollout_reward_max = -math.inf
        self._reset_iteration_statistics()
        self._last_metrics = dict(train_metrics)
        if self.nonfinite_recorder is not None:
            self.nonfinite_recorder.flush_summary(status="RUNNING")
        self._flush_metrics()
        completed_rollouts = int(iteration) + 1
        if (
            self.checkpoint_interval > 0
            and completed_rollouts % self.checkpoint_interval == 0
        ):
            checkpoint_name = (
                f"checkpoint_{self.rollout_offset + completed_rollouts:04d}.pt"
            )
            checkpoint_path = self._save_checkpoint(checkpoint_name)
            self.metrics.setdefault("periodic_checkpoints", []).append(
                {
                    "rollout_index": self.rollout_offset + completed_rollouts,
                    "path": str(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                }
            )
            self._flush_metrics()
        print(
            f"{self.audit_label}_UPDATE phase={self.phase} iteration={iteration} "
            f"optimizer_steps={self.optimizer_step_counts} metrics={_jsonable(train_metrics)}",
            flush=True,
        )

    def _run_contract_checks(self) -> dict[str, Any]:
        if self._last_fixed_input is None:
            raise AssertionError("no rollout observation available for contract checks")
        td = self._last_fixed_input.clone().to(self.policy.device)
        policy_obs = td["policy"]
        if policy_obs.shape[-1] != EXPECTED_OBSERVATION_DIM:
            raise AssertionError("policy observation contract changed")
        _assert_finite("contract.policy", policy_obs)

        if self.phase == "train":
            with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
                self.policy.encoder_priv(td)
                teacher_latent = td["priv_feature"].clone()
                self.policy.actor_teacher(td)
            if teacher_latent.shape[-1] != EXPECTED_LATENT_DIM:
                raise AssertionError("Teacher latent width changed")
            if td["_actor_inp"].shape[-1] != EXPECTED_ACTOR_INPUT_DIM:
                raise AssertionError("Teacher actor input width changed")
            if td["action"].shape[-1] != EXPECTED_ACTION_DIM:
                raise AssertionError("Teacher action width changed")
            if "student_priv_feature" in td.keys():
                raise AssertionError("Teacher path unexpectedly ran the Student quantizer")
            torch.testing.assert_close(td["_actor_inp"][..., -40:], teacher_latent)
            for label, tensor in (
                ("teacher_latent", teacher_latent),
                ("teacher_actor_input", td["_actor_inp"]),
                ("teacher_action", td["action"]),
            ):
                _assert_finite(label, tensor)
            return {
                "policy_observation_shape": list(policy_obs.shape),
                "teacher_latent_shape": list(teacher_latent.shape),
                "teacher_actor_input_shape": list(td["_actor_inp"].shape),
                "teacher_action_shape": list(td["action"].shape),
                "teacher_latent_continuous": True,
                "student_quantizer_ran_on_teacher_path": False,
                "pass": True,
            }

        self.policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.policy.encoder_priv(td)
        teacher_before = td["priv_feature"].detach().clone()
        self.policy.adapt_module(td)
        priv_pred = td["priv_pred"]
        priv_pred.retain_grad()
        self.policy.student_latent_quantizer(td)
        quantized = td["student_priv_feature"]
        teacher_after = td["priv_feature"].detach().clone()
        with set_exploration_type(ExplorationType.MODE):
            self.policy.actor_student(td)

        expected_shapes = {
            "priv_pred": EXPECTED_LATENT_DIM,
            "quantized": EXPECTED_LATENT_DIM,
            "actor_input": EXPECTED_ACTOR_INPUT_DIM,
            "action": EXPECTED_ACTION_DIM,
        }
        actual = {
            "priv_pred": priv_pred,
            "quantized": quantized,
            "actor_input": td["_actor_inp"],
            "action": td["action"],
        }
        for name, tensor in actual.items():
            if tensor.shape[-1] != expected_shapes[name]:
                raise AssertionError(
                    f"{name} width {tensor.shape[-1]} != {expected_shapes[name]}"
                )
            _assert_finite(f"contract.{name}", tensor)
        quant_min = float(quantized.min().item())
        quant_max = float(quantized.max().item())
        if quant_min < -4.0 or quant_max > 4.0:
            raise AssertionError(f"quantized range [{quant_min}, {quant_max}] exceeds [-4, 4]")
        torch.testing.assert_close(quantized, quantized.round())
        torch.testing.assert_close(teacher_after, teacher_before)
        torch.testing.assert_close(td["_actor_inp"][..., -40:], quantized)

        valid = torch.ones_like(priv_pred[..., :1], dtype=torch.bool)
        continuous_estimator_loss = student_estimator_mse(
            priv_pred, teacher_before, valid
        )
        _assert_finite("continuous_estimator_loss", continuous_estimator_loss)
        quantized_alignment_loss = torch.mean(
            torch.nn.functional.mse_loss(
                quantized, teacher_before, reduction="none"
            )
            * valid
        )
        _assert_finite("quantized_alignment_loss", quantized_alignment_loss)

        quantized.sum().backward()
        ste_grad = priv_pred.grad
        if ste_grad is None:
            raise AssertionError("STE probe produced no gradient for continuous priv_pred")
        _assert_finite("ste_gradient", ste_grad)
        ste_nonzero = int(torch.count_nonzero(ste_grad).item())
        if ste_nonzero == 0:
            raise AssertionError("STE probe gradient is zero")
        ste_adapt_gradient = gradient_report(self.policy.adapt_module)
        if (
            not ste_adapt_gradient["finite"]
            or ste_adapt_gradient["nonzero_gradient_tensors"] == 0
        ):
            raise AssertionError(
                f"STE did not reach adapt_module: {ste_adapt_gradient}"
            )

        report = {
            "policy_observation_shape": list(policy_obs.shape),
            "priv_pred_shape": list(priv_pred.shape),
            "quantized_latent_shape": list(quantized.shape),
            "quantized_min": quant_min,
            "quantized_max": quant_max,
            "quantized_levels": sorted(
                float(value) for value in torch.unique(quantized.detach()).cpu().tolist()
            ),
            "quantized_tensor_finite": True,
            "actor_input_shape": list(td["_actor_inp"].shape),
            "action_shape": list(td["action"].shape),
            "actor_input_uses_quantized_latent": True,
            "teacher_latent_unchanged_by_quantizer": True,
            "quantizer_trainable_parameter_count": len(
                list(self.policy.student_latent_quantizer.parameters())
            ),
            "estimator_target": "continuous priv_pred vs continuous teacher priv_feature",
            "continuous_estimator_loss": float(continuous_estimator_loss.detach().item()),
            "quantized_alignment_loss_diagnostic_only": float(
                quantized_alignment_loss.detach().item()
            ),
            "ste_priv_pred_gradient_norm": float(
                ste_grad.detach().double().norm().item()
            ),
            "ste_priv_pred_nonzero_elements": ste_nonzero,
            "ste_adapt_module_gradient": ste_adapt_gradient,
            "pass": True,
        }
        self._write_json("quantizer_report.json", report)
        return report

    def _deterministic_action(self, policy) -> torch.Tensor:
        td = self._last_fixed_input.clone().to(policy.device)
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            policy.get_rollout_policy("checkpoint_roundtrip")(td)
        action = td["action"].detach().cpu()
        _assert_finite("roundtrip_action", action)
        return action

    def _checkpoint_payload(self) -> OrderedDict:
        payload = OrderedDict()
        payload["wandb"] = {"name": None, "id": None, "mode": "disabled"}
        payload["policy"] = self.policy.state_dict()
        payload["env"] = self.env.state_dict()
        payload["cfg"] = self.cfg
        payload["vecnorm"] = self.vecnorm.state_dict()
        return payload

    def _save_checkpoint(self, file_name: str, *, synchronize: bool = True) -> Path:
        checkpoint_path = self.root_output_dir / file_name
        if aa.is_main_process():
            torch.save(self._checkpoint_payload(), checkpoint_path)
        if synchronize and dist.is_available() and dist.is_initialized():
            dist.barrier()
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise AssertionError(
                f"checkpoint save produced an empty file: {checkpoint_path}"
            )
        return checkpoint_path

    def abort(self, error: BaseException) -> Path | None:
        """Persist the recoverable state and diagnostics before propagating failure."""
        checkpoint_path = None
        checkpoint_error = None
        try:
            if aa.is_main_process():
                checkpoint_path = self._save_checkpoint(
                    "checkpoint_abort.pt", synchronize=False
                )
            (self.output_dir / "checkpoint_path.txt").write_text(
                str(checkpoint_path) + "\n", encoding="utf-8"
            )
        except BaseException as save_error:  # preserve the original failure
            checkpoint_error = f"{type(save_error).__name__}: {save_error}"

        nonfinite_summary = None
        if self.nonfinite_recorder is not None:
            nonfinite_summary = self.nonfinite_recorder.flush_summary(
                status="ABORTED"
            )
        self.metrics.update(
            {
                "status": "ABORTED",
                "rollout_count": len(self.metrics.get("rollouts", [])),
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "abort_checkpoint_path": (
                    None if checkpoint_path is None else str(checkpoint_path)
                ),
                "abort_checkpoint_error": checkpoint_error,
                "nonfinite_summary": nonfinite_summary,
            }
        )
        self._flush_metrics()
        print(
            f"{self.audit_label}_RESULT=ABORTED "
            f"error={type(error).__name__}: {error} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )
        return checkpoint_path

    def _verify_final_module_changes(self) -> dict[str, Any]:
        report = {}
        for name in self._MODULE_NAMES:
            delta = parameter_delta(
                self._initial_parameters[name], getattr(self.policy, name)
            )
            expected_changed = name in self._EXPECTED_CHANGED[self.phase]
            expected_frozen = name in self._EXPECTED_FROZEN[self.phase]
            delta["expected_changed"] = expected_changed
            delta["expected_frozen"] = expected_frozen
            if expected_changed and not delta["changed"]:
                raise AssertionError(f"trained module {name} never changed")
            if expected_frozen and delta["changed"]:
                raise AssertionError(f"phase-frozen module {name} changed: {delta}")
            report[name] = delta
        return report

    def _reload_checkpoint(self, checkpoint_path: Path, expected_action: torch.Tensor):
        from scripts.utils.helpers import make_env_policy

        reload_cfg = OmegaConf.create(
            copy.deepcopy(OmegaConf.to_container(self.cfg, resolve=False))
        )
        reload_cfg.checkpoint_path = str(checkpoint_path)
        if reload_cfg.task.get("nonfinite_diagnostics", None) is not None:
            reload_cfg.task.nonfinite_diagnostics.enabled = False
        reload_env = None
        caught_warnings = []
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                reload_env, reload_policy, _, _ = make_env_policy(reload_cfg)
            caught_warnings = [str(record.message) for record in records]
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            policy_state = checkpoint["policy"]
            key_reports = {}
            module_deltas = {}
            for name, module in {
                **{key: getattr(reload_policy, key) for key in self._MODULE_NAMES},
                "student_latent_quantizer": reload_policy.student_latent_quantizer,
            }.items():
                saved = policy_state.get(name, {})
                key_reports[name] = state_key_report(saved, module)
                if self.phase == "train" and name == "actor_student":
                    saved = policy_state["actor_teacher"]
                module_deltas[name] = _module_state_max_abs_delta(saved, module)
            critical = {
                name: report
                for name, report in key_reports.items()
                if report["missing_keys"] or report["unexpected_keys"]
            }
            if critical:
                raise AssertionError(f"reload key mismatch: {critical}")
            if any(
                not math.isfinite(delta) or delta != 0.0
                for delta in module_deltas.values()
            ):
                raise AssertionError(f"reload tensor mismatch: {module_deltas}")

            loaded_action = self._deterministic_action(reload_policy)
            torch.testing.assert_close(
                loaded_action, expected_action, rtol=1.0e-5, atol=1.0e-6
            )
            action_max_abs_delta = float(
                (loaded_action - expected_action).abs().max().item()
            )
            report = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "saved_phase": policy_state.get("last_phase"),
                "reloaded_phase": str(reload_policy.cfg.phase),
                "module_key_reports": key_reports,
                "critical_missing_or_unexpected": critical,
                "module_max_abs_delta": module_deltas,
                "warnings": caught_warnings,
                "action_shape": list(loaded_action.shape),
                "action_rtol": 1.0e-5,
                "action_atol": 1.0e-6,
                "action_max_abs_delta": action_max_abs_delta,
                "action_matches": True,
                "pass": True,
            }
            return report
        finally:
            if reload_env is not None:
                reload_env.close(raise_if_closed=False)
            gc.collect()
            torch.cuda.empty_cache()

    def finalize(self) -> Path:
        minimum_rollouts = int(self.audit_cfg.get("minimum_rollout_iterations", 2))
        if len(self.metrics["rollouts"]) < minimum_rollouts:
            raise AssertionError(
                f"fewer than {minimum_rollouts} rollout updates completed"
            )
        contract_report = self._run_contract_checks()
        final_changes = self._verify_final_module_changes()
        self._write_json("gradient_report.json", self._gradient_reports)

        ppo_steps = (
            self.total_iters
            * int(self.cfg.algo.ppo_epochs)
            * int(self.cfg.algo.num_minibatches)
        )
        estimator_steps = self.total_iters * 2 * int(self.cfg.algo.num_minibatches)
        expected_counts = {
            "train": {
                "teacher": ppo_steps,
                "student": 0,
                "critic": ppo_steps,
                "estimator": estimator_steps,
            },
            "adapt": {
                "teacher": 0,
                "student": 0,
                "critic": 0,
                "estimator": estimator_steps,
            },
            "finetune": {
                "teacher": 0,
                "student": ppo_steps,
                "critic": ppo_steps,
                "estimator": 0,
            },
        }[self.phase]
        if self.optimizer_step_counts != expected_counts:
            raise AssertionError(
                f"optimizer step counts {self.optimizer_step_counts} != {expected_counts}"
            )

        device = str(self.base_env.device)
        training_resource_peaks = {
            "gpu_allocated_mib": int(torch.cuda.max_memory_allocated(device) // MIB),
            "gpu_reserved_mib": int(torch.cuda.max_memory_reserved(device) // MIB),
            "cpu_peak_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            / 1024.0,
        }
        expected_action = self._deterministic_action(self.policy)
        checkpoint_path = self._save_checkpoint("checkpoint_final.pt")
        (self.output_dir / "checkpoint_path.txt").write_text(
            str(checkpoint_path) + "\n", encoding="utf-8"
        )
        verify_reload = bool(self.audit_cfg.get("verify_checkpoint_reload", True))
        reload_report = None
        if verify_reload:
            if self.world_size != 1:
                raise RuntimeError(
                    "in-process checkpoint reload is disabled for DDP; run the "
                    "post-training single-GPU roundtrip verifier"
                )
            reload_report = self._reload_checkpoint(checkpoint_path, expected_action)
            self._write_json("checkpoint_roundtrip_report.json", reload_report)

        self._sample_memory("after_checkpoint_reload")
        timing_rows = [row["timing"] for row in self.metrics["rollouts"] if row["timing"]]
        timing_summary = {}
        if timing_rows:
            total_frames = sum(row["batch_size"] for row in self.metrics["rollouts"])
            total_rollout_time = sum(row["rollout_time_s"] for row in timing_rows)
            total_training_time = sum(row["training_time_s"] for row in timing_rows)
            timing_summary = {
                "total_rollout_time_s": total_rollout_time,
                "total_training_time_s": total_training_time,
                "rollout_steps_per_second": total_frames / total_rollout_time,
                "end_to_end_steps_per_second": total_frames
                / (total_rollout_time + total_training_time),
            }
        sampled_gpu_peak_mib = max(
            sample["used_mib"] for sample in self.metrics["gpu_memory_samples"]
        )
        cpu_peak_rss_mib = max(
            sample["cpu_peak_rss_mib"] for sample in self.metrics["gpu_memory_samples"]
        )
        cpu_current_samples = [
            sample["cpu_current_rss_mib"]
            for sample in self.metrics["gpu_memory_samples"]
            if sample.get("cpu_current_rss_mib") is not None
        ]
        nonfinite_summary = None
        if self.nonfinite_recorder is not None:
            nonfinite_summary = self.nonfinite_recorder.flush_summary(status="PASS")
        self.metrics.update(
            {
                "status": "PASS",
                "rollout_count": len(self.metrics["rollouts"]),
                "optimizer_step_counts": dict(self.optimizer_step_counts),
                "nonfinite_state_recoveries_total": int(
                    self.base_env.nonfinite_state_recovery_count
                ),
                "last_train_metrics": _jsonable(self._last_metrics),
                "contract": contract_report,
                "final_parameter_changes": final_changes,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "checkpoint_reload": reload_report,
                "checkpoint_reload_deferred": not verify_reload,
                "timing": timing_summary,
                "training_resource_peaks": training_resource_peaks,
                "gpu": {
                    "peak_allocated_mib": int(
                        torch.cuda.max_memory_allocated(device) // MIB
                    ),
                    "peak_reserved_mib": int(
                        torch.cuda.max_memory_reserved(device) // MIB
                    ),
                    "sampled_device_used_peak_mib": sampled_gpu_peak_mib,
                },
                "cpu": {"peak_rss_mib": cpu_peak_rss_mib},
                "cpu_current_rss_samples_mib": cpu_current_samples,
                "motion_coverage": {
                    "dataset_motion_count": self._motion_count,
                    "sampled_unique_motion_count": int(
                        self._sampled_motion_ids.sum().item()
                    ),
                    "sampled_motion_ids": self._sampled_motion_ids.nonzero()
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .tolist(),
                },
                "nonfinite_summary": nonfinite_summary,
                "no_nan_or_inf": True,
            }
        )
        self._flush_metrics()
        print(
            f"{self.audit_label}_RESULT=PASS phase={self.phase} "
            f"checkpoint={checkpoint_path} optimizer_steps={self.optimizer_step_counts}",
            flush=True,
        )
        return checkpoint_path
