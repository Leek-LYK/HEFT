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
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.envs.utils import ExplorationType, set_exploration_type

from active_adaptation.learning.ppo.ppo import student_estimator_mse


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
    if values["world_size"] != 1:
        raise ValueError("PPO smoke must run with world_size=1 (no DDP)")
    if not 1 <= values["num_envs"] <= 256:
        raise ValueError("PPO smoke num_envs must be in [1, 256]")
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
    if values["total_frames"] <= 0 or values["total_frames"] % batch_size:
        raise ValueError("total_frames must be a positive multiple of the rollout batch")

    rollout_updates = values["total_frames"] // batch_size
    if not 2 <= rollout_updates <= 5:
        raise ValueError("PPO smoke must execute between 2 and 5 rollout updates")
    return {
        "batch_size": batch_size,
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


class PPOSmokeDiagnostics:
    """Observe and attest a short real PPO phase without changing its algorithm."""

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
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.vecnorm = vecnorm
        self.base_env = env.base_env
        self.phase = str(cfg.algo.phase)
        if self.phase not in self._EXPECTED_CHANGED:
            raise ValueError(f"unsupported PPO smoke phase: {self.phase}")

        if os.environ.get("WORLD_SIZE", "1") != "1":
            raise RuntimeError("PPO smoke forbids WORLD_SIZE other than 1")
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices != "3":
            raise RuntimeError(
                f"PPO smoke requires CUDA_VISIBLE_DEVICES=3, got {visible_devices!r}"
            )
        if str(self.base_env.device) != "cuda:0":
            raise RuntimeError(
                f"PPO smoke expects logical cuda:0, got {self.base_env.device}"
            )

        smoke_cfg = cfg.get("smoke", {})
        output_dir = Path(str(smoke_cfg.get("output_dir", ""))).resolve()
        smoke_root_raw = os.environ.get("PPO_SMOKE_ROOT")
        if not smoke_root_raw:
            raise RuntimeError("PPO_SMOKE_ROOT must be set for PPO smoke")
        smoke_root = Path(smoke_root_raw).resolve()
        if output_dir.parent != smoke_root or output_dir.name != self.phase:
            raise RuntimeError(
                f"smoke output_dir must be {smoke_root / self.phase}, got {output_dir}"
            )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        schedule = validate_smoke_schedule(
            num_envs=self.base_env.num_envs,
            rollout_horizon=cfg.algo.train_every,
            num_minibatches=cfg.algo.num_minibatches,
            total_frames=cfg.total_frames,
            world_size=1,
        )
        if schedule["rollout_updates"] != int(total_iters):
            raise AssertionError(
                f"derived rollout updates {schedule['rollout_updates']} != train.py {total_iters}"
            )
        if int(cfg.algo.ppo_epochs) != 1:
            raise ValueError("PPO smoke expects exactly one PPO learning epoch")
        if bool(cfg.eval_render) or bool(cfg.app.enable_cameras):
            raise RuntimeError("PPO smoke forbids rendering and cameras")
        if not bool(cfg.headless) or not bool(cfg.app.headless):
            raise RuntimeError("PPO smoke must remain headless")
        if str(cfg.wandb.mode) != "disabled":
            raise RuntimeError("PPO smoke must keep W&B disabled")
        if bool(cfg.train_record.enabled):
            raise RuntimeError("PPO smoke must keep train_record disabled")

        dataset = self.base_env.command_manager.dataset
        teacher_dataset = dataset.teacher_dataset
        mem_root = Path(os.environ["MEMPATH"]).resolve()
        loaded_paths = []
        for raw_path in teacher_dataset.dataset_paths:
            path = Path(raw_path)
            loaded_paths.append((path if path.is_absolute() else mem_root / path).resolve())
        expected_dataset = (mem_root / EXPECTED_DATASET_MEM_PATH).resolve()
        if loaded_paths != [expected_dataset]:
            raise AssertionError(
                f"PPO smoke loaded datasets {loaded_paths}, expected {[expected_dataset]}"
            )
        if str(dataset.ds_device) != "cpu" or str(teacher_dataset.ds_device) != "cpu":
            raise AssertionError("PPO smoke dataset and cache must stay on CPU")
        if int(self.base_env.action_dim) != EXPECTED_ACTION_DIM:
            raise AssertionError(
                f"action dim {self.base_env.action_dim} != {EXPECTED_ACTION_DIM}"
            )
        if int(cfg.algo.latent_dim) != EXPECTED_LATENT_DIM:
            raise AssertionError(
                f"latent dim {cfg.algo.latent_dim} != {EXPECTED_LATENT_DIM}"
            )
        if not bool(policy.student_latent_quantizer_enabled):
            raise AssertionError("Roban PPO smoke requires the Student quantizer")
        if list(policy.student_latent_quantizer.parameters()):
            raise AssertionError("Student quantizer unexpectedly has trainable parameters")

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
        self._last_fixed_input = None
        self._last_metrics: dict[str, Any] = {}
        self._gradient_reports: dict[str, Any] = {}
        self._phase_load_report = self._verify_input_checkpoint()
        if self._phase_load_report is not None:
            self._write_json("checkpoint_load_report.json", self._phase_load_report)

        self.metrics: dict[str, Any] = {
            "status": "RUNNING",
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
                "optimizer": str(cfg.algo.optimizer),
                "dataset_path": str(expected_dataset),
                "dataset_mem_path": EXPECTED_DATASET_MEM_PATH,
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
            "PPO_SMOKE_START "
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
            if raw_path is not None:
                raise AssertionError("train smoke must start without a checkpoint")
            return None
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

    def _sample_memory(self, label: str) -> dict[str, int | str]:
        device = str(self.base_env.device)
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        sample = {
            "label": label,
            "used_mib": int((total_bytes - free_bytes) // MIB),
            "free_mib": int(free_bytes // MIB),
            "torch_allocated_mib": int(torch.cuda.memory_allocated(device) // MIB),
            "torch_reserved_mib": int(torch.cuda.memory_reserved(device) // MIB),
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

    def observe_transition(self, td, *, iteration: int, rollout_step: int) -> None:
        observation = td["policy"]
        action = td["action"]
        reward = td["next", "reward"]
        value = td["state_value"]
        next_value = td["next", "state_value"]
        if tuple(observation.shape) != (
            int(self.base_env.num_envs),
            EXPECTED_OBSERVATION_DIM,
        ):
            raise AssertionError(f"policy observation shape mismatch: {observation.shape}")
        if tuple(action.shape) != (int(self.base_env.num_envs), EXPECTED_ACTION_DIM):
            raise AssertionError(f"action shape mismatch: {action.shape}")
        for label, tensor in (
            ("policy_observation", observation),
            ("action", action),
            ("reward", reward),
            ("value", value),
            ("next_value", next_value),
            ("qpos", self.base_env.sim.data.qpos),
            ("qvel", self.base_env.sim.data.qvel),
            ("torque", self.base_env.scene["robot"].data.qfrc_actuator),
        ):
            _assert_finite(label, tensor)
        actor_input = td.get("_actor_inp", None)
        if actor_input is not None:
            if actor_input.shape[-1] != EXPECTED_ACTOR_INPUT_DIM:
                raise AssertionError(
                    f"actor input width {actor_input.shape[-1]} != {EXPECTED_ACTOR_INPUT_DIM}"
                )
            _assert_finite("actor_input", actor_input)
        self._rollout_reward_min = min(
            self._rollout_reward_min, float(reward.min().item())
        )
        self._rollout_reward_max = max(
            self._rollout_reward_max, float(reward.max().item())
        )
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

    def after_update(self, data, train_metrics: Mapping[str, Any], *, iteration: int) -> None:
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

        rollout = {
            "iteration": int(iteration),
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
            "gpu_memory": self._sample_memory(f"iteration_{iteration}_after_update"),
        }
        self.metrics["rollouts"].append(rollout)
        self.metrics["optimizer_step_counts"] = dict(self.optimizer_step_counts)
        self._rollout_reward_min = math.inf
        self._rollout_reward_max = -math.inf
        self._last_metrics = dict(train_metrics)
        self._flush_metrics()
        print(
            f"PPO_SMOKE_UPDATE phase={self.phase} iteration={iteration} "
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
        if len(self.metrics["rollouts"]) < 2:
            raise AssertionError("fewer than two rollout updates completed")
        contract_report = self._run_contract_checks()
        final_changes = self._verify_final_module_changes()
        self._write_json("gradient_report.json", self._gradient_reports)

        expected_counts = {
            "train": {"teacher": 2, "student": 0, "critic": 2, "estimator": 4},
            "adapt": {"teacher": 0, "student": 0, "critic": 0, "estimator": 4},
            "finetune": {"teacher": 0, "student": 2, "critic": 2, "estimator": 0},
        }[self.phase]
        if self.optimizer_step_counts != expected_counts:
            raise AssertionError(
                f"optimizer step counts {self.optimizer_step_counts} != {expected_counts}"
            )

        expected_action = self._deterministic_action(self.policy)
        checkpoint_path = self.output_dir / "checkpoint_final.pt"
        torch.save(self._checkpoint_payload(), checkpoint_path)
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise AssertionError("checkpoint save produced an empty file")
        (self.output_dir / "checkpoint_path.txt").write_text(
            str(checkpoint_path) + "\n", encoding="utf-8"
        )
        reload_report = self._reload_checkpoint(checkpoint_path, expected_action)
        self._write_json("checkpoint_roundtrip_report.json", reload_report)

        self._sample_memory("after_checkpoint_reload")
        device = str(self.base_env.device)
        self.metrics.update(
            {
                "status": "PASS",
                "rollout_count": len(self.metrics["rollouts"]),
                "optimizer_step_counts": dict(self.optimizer_step_counts),
                "last_train_metrics": _jsonable(self._last_metrics),
                "contract": contract_report,
                "final_parameter_changes": final_changes,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "checkpoint_reload": reload_report,
                "gpu": {
                    "peak_allocated_mib": int(
                        torch.cuda.max_memory_allocated(device) // MIB
                    ),
                    "peak_reserved_mib": int(
                        torch.cuda.max_memory_reserved(device) // MIB
                    ),
                },
                "no_nan_or_inf": True,
            }
        )
        self._flush_metrics()
        print(
            f"PPO_SMOKE_RESULT=PASS phase={self.phase} "
            f"checkpoint={checkpoint_path} optimizer_steps={self.optimizer_step_counts}",
            flush=True,
        )
        return checkpoint_path
