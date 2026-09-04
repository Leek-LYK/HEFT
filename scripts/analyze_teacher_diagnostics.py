#!/usr/bin/env python3
"""Aggregate Teacher diagnostic windows and evaluate the 200-rollout gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


TRAIN_METRIC_KEYS = (
    "actor/policy_loss",
    "critic/value_loss",
    "actor/entropy",
    "actor/kl",
    "actor/actor_grad_norm",
    "action/encoder_grad_norm",
    "critic/critic_grad_norm",
)
BODY_Z_KEY = "termination/body_z_termination"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return fmean(finite) if finite else None


def _slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2.0
    y_mean = fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator == 0.0:
        return 0.0
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def aggregate_window(
    rows: list[dict[str, Any]], start: int, end: int
) -> dict[str, Any]:
    selected = [
        row for row in rows if start <= int(row["rollout_index"]) <= end
    ]
    stat_sums: defaultdict[str, float] = defaultdict(float)
    episodes = 0
    terminated = 0
    truncated = 0
    immediate = 0
    episode_length_sum = 0.0
    for row in selected:
        episode = row["episode"]
        episodes += int(episode.get("completed_episodes", 0))
        terminated += int(episode.get("terminated_episodes", 0))
        truncated += int(episode.get("truncated_episodes", 0))
        immediate += int(episode.get("immediate_terminations_le_10_steps", 0))
        episode_length_sum += float(episode.get("episode_length_sum", 0.0))
        for key, value in episode.get("episode_stat_sums", {}).items():
            stat_sums[str(key)] += float(value)

    per_episode = {
        key: value / episodes for key, value in sorted(stat_sums.items())
    } if episodes else {}
    per_step = {
        key: value / episode_length_sum
        for key, value in sorted(stat_sums.items())
        if key != "episode_len"
    } if episode_length_sum > 0.0 else {}
    reward_returns = [
        value for key, value in per_episode.items() if key.endswith("/return")
    ]

    metric_means = {
        key: _mean(
            row.get("metrics", {}).get(key)
            for row in selected
            if row.get("metrics", {}).get(key) is not None
        )
        for key in TRAIN_METRIC_KEYS
    }
    gpu_used = [float(row["gpu_memory"]["used_mib"]) for row in selected]
    gpu_reserved = [
        float(row["gpu_memory"]["torch_reserved_mib"]) for row in selected
    ]
    cpu_current = [
        float(row["gpu_memory"]["cpu_current_rss_mib"])
        for row in selected
        if row["gpu_memory"].get("cpu_current_rss_mib") is not None
    ]
    total_frames = sum(int(row.get("batch_size", 0)) for row in selected)
    rollout_seconds = sum(
        float(row.get("timing", {}).get("rollout_time_s", 0.0))
        for row in selected
    )
    training_seconds = sum(
        float(row.get("timing", {}).get("training_time_s", 0.0))
        for row in selected
    )
    event_count = sum(
        int(row.get("nonfinite_events_in_iteration", 0)) for row in selected
    )
    decimation = 4
    if selected:
        batch_size = int(selected[0].get("batch_size", 0))
        transition_count = batch_size * len(selected)
    else:
        transition_count = 0

    return {
        "name": f"rollout_{start}_{end}",
        "start_rollout": start,
        "end_rollout": end,
        "rollouts_present": len(selected),
        "frames": total_frames,
        "completed_episodes": episodes,
        "mean_episode_reward": sum(reward_returns) if reward_returns else None,
        "mean_episode_length": (
            episode_length_sum / episodes if episodes else None
        ),
        "body_z_termination_ratio": (
            stat_sums.get(BODY_Z_KEY, 0.0) / episodes if episodes else None
        ),
        "termination_ratio": terminated / episodes if episodes else None,
        "truncation_ratio": truncated / episodes if episodes else None,
        "termination_le_10_step_ratio": immediate / episodes if episodes else None,
        "mean_episode_stats": per_episode,
        "mean_episode_stats_per_step": per_step,
        "tracking_return_per_episode": per_episode.get("tracking/return"),
        "loco_return_per_episode": per_episode.get("loco/return"),
        "training_metrics": metric_means,
        "throughput": {
            "rollout_steps_per_second": (
                total_frames / rollout_seconds if rollout_seconds > 0.0 else None
            ),
            "total_steps_per_second": (
                total_frames / (rollout_seconds + training_seconds)
                if rollout_seconds + training_seconds > 0.0
                else None
            ),
        },
        "memory": {
            "gpu_used_mean_mib": _mean(gpu_used),
            "gpu_used_max_mib": max(gpu_used, default=None),
            "gpu_used_slope_mib_per_rollout": _slope(gpu_used),
            "torch_reserved_mean_mib": _mean(gpu_reserved),
            "torch_reserved_max_mib": max(gpu_reserved, default=None),
            "torch_reserved_slope_mib_per_rollout": _slope(gpu_reserved),
            "cpu_rss_mean_mib": _mean(cpu_current),
            "cpu_rss_max_mib": max(cpu_current, default=None),
            "cpu_rss_slope_mib_per_rollout": _slope(cpu_current),
        },
        "nonfinite": {
            "events": event_count,
            "policy_transitions": transition_count,
            "physics_substeps": transition_count * decimation,
            "event_rate_per_policy_transition": (
                event_count / transition_count if transition_count else None
            ),
            "event_rate_per_physics_substep": (
                event_count / (transition_count * decimation)
                if transition_count
                else None
            ),
        },
    }


def _criterion(value: Any, passed: bool, detail: str) -> dict[str, Any]:
    return {"pass": bool(passed), "value": value, "detail": detail}


def evaluate_200_gate(
    metrics: dict[str, Any],
    summary: dict[str, Any],
    first: dict[str, Any],
    last: dict[str, Any],
) -> dict[str, Any]:
    criteria: dict[str, dict[str, Any]] = {}
    criteria["finite_loss_and_gradients"] = _criterion(
        metrics.get("no_nan_or_inf"),
        metrics.get("status") == "PASS" and metrics.get("no_nan_or_inf") is True,
        "diagnostic run completed with every loss, metric, and gradient finite",
    )
    criteria["no_cuda_oom"] = _criterion(
        metrics.get("failure"),
        metrics.get("status") == "PASS"
        and "out of memory" not in str(metrics.get("failure", "")).lower(),
        "no CUDA OOM was recorded",
    )
    pre_control = int(summary.get("pre_step_control_nonfinite_events", 0))
    criteria["finite_pre_step_action_target_ctrl"] = _criterion(
        pre_control,
        pre_control == 0,
        "sim.step inputs action/joint_target/ctrl stayed finite",
    )
    event_rate = float(summary.get("event_rate_per_policy_transition", math.inf))
    criteria["nonfinite_event_rate"] = _criterion(
        event_rate, event_rate <= 1.0e-4, "required <= 1e-4"
    )
    max_repeat = int(summary.get("max_motion_frame_repeat", 0))
    criteria["motion_frame_repeat"] = _criterion(
        max_repeat, max_repeat <= 10, "required <= 10"
    )
    max_iteration = int(summary.get("max_events_in_iteration", 0))
    criteria["events_per_iteration"] = _criterion(
        max_iteration, max_iteration <= 20, "required <= 20"
    )
    max_burst = int(summary.get("max_contiguous_event_burst", 0))
    first_event_rate = first["nonfinite"]["event_rate_per_policy_transition"] or 0.0
    last_event_rate = last["nonfinite"]["event_rate_per_policy_transition"] or 0.0
    no_growth = last_event_rate <= max(1.0e-4, 3.0 * first_event_rate)
    criteria["no_explosive_event_burst"] = _criterion(
        {"max_contiguous_burst": max_burst, "first_rate": first_event_rate, "last_rate": last_event_rate},
        max_burst <= 20 and no_growth,
        "contiguous burst <= 20 and last-window rate is not explosively higher",
    )

    rows = metrics.get("rollouts", [])
    warm_rows = rows[min(10, len(rows)) :]
    gpu_reserved = [
        float(row["gpu_memory"]["torch_reserved_mib"]) for row in warm_rows
    ]
    cpu_rss = [
        float(row["gpu_memory"]["cpu_current_rss_mib"])
        for row in warm_rows
        if row["gpu_memory"].get("cpu_current_rss_mib") is not None
    ]
    gpu_slope = _slope(gpu_reserved) or 0.0
    cpu_slope = _slope(cpu_rss) or 0.0
    gpu_half = max(1, len(gpu_reserved) // 2)
    cpu_half = max(1, len(cpu_rss) // 2)
    gpu_delta = (_mean(gpu_reserved[gpu_half:]) or 0.0) - (_mean(gpu_reserved[:gpu_half]) or 0.0)
    cpu_delta = (_mean(cpu_rss[cpu_half:]) or 0.0) - (_mean(cpu_rss[:cpu_half]) or 0.0)
    criteria["gpu_memory_stable"] = _criterion(
        {"slope_mib_per_rollout": gpu_slope, "late_minus_early_mean_mib": gpu_delta},
        gpu_slope <= 1.0 and gpu_delta <= 128.0,
        "after warmup: slope <= 1 MiB/rollout and late mean growth <= 128 MiB",
    )
    criteria["cpu_rss_stable"] = _criterion(
        {"slope_mib_per_rollout": cpu_slope, "late_minus_early_mean_mib": cpu_delta},
        cpu_slope <= 1.0 and cpu_delta <= 128.0,
        "after warmup: slope <= 1 MiB/rollout and late mean growth <= 128 MiB",
    )
    reload_report = metrics.get("checkpoint_reload", {})
    criteria["checkpoint_roundtrip"] = _criterion(
        reload_report,
        reload_report.get("pass") is True
        and reload_report.get("action_matches") is True,
        "checkpoint save/reload and deterministic action must match",
    )

    reward_ratio = _ratio(last.get("mean_episode_reward"), first.get("mean_episode_reward"))
    length_ratio = _ratio(last.get("mean_episode_length"), first.get("mean_episode_length"))
    criteria["reward_retention"] = _criterion(
        reward_ratio,
        reward_ratio is not None and reward_ratio >= 0.8,
        "last 50 mean reward / first 50 mean reward >= 0.8",
    )
    criteria["episode_length_retention"] = _criterion(
        length_ratio,
        length_ratio is not None and length_ratio >= 0.8,
        "last 50 mean episode length / first 50 mean episode length >= 0.8",
    )

    first_metrics = first["training_metrics"]
    last_metrics = last["training_metrics"]
    kl_ok = (last_metrics["actor/kl"] or math.inf) <= max(
        0.05, 5.0 * (first_metrics["actor/kl"] or 0.0)
    )
    value_ok = (last_metrics["critic/value_loss"] or math.inf) <= max(
        1.0, 5.0 * (first_metrics["critic/value_loss"] or 0.0)
    )
    entropy_first = first_metrics["actor/entropy"]
    entropy_last = last_metrics["actor/entropy"]
    entropy_ok = (
        entropy_first is not None
        and entropy_last is not None
        and abs(entropy_last - entropy_first)
        <= max(5.0, 0.25 * abs(entropy_first))
    )
    grad_ok = True
    grad_ratios = {}
    for key in (
        "actor/actor_grad_norm",
        "action/encoder_grad_norm",
        "critic/critic_grad_norm",
    ):
        ratio = _ratio(last_metrics[key], first_metrics[key])
        grad_ratios[key] = ratio
        grad_ok = grad_ok and ratio is not None and ratio <= 5.0
    criteria["ppo_metrics_stable"] = _criterion(
        {
            "first": first_metrics,
            "last": last_metrics,
            "gradient_ratios": grad_ratios,
        },
        kl_ok and value_ok and entropy_ok and grad_ok,
        "KL/value/entropy remain bounded and each gradient-norm ratio <= 5",
    )
    return {
        "status": "PASS" if all(row["pass"] for row in criteria.values()) else "FAIL",
        "criteria": criteria,
        "first_50": first,
        "last_50": last,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--continuation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-gate-pass", action="store_true")
    args = parser.parse_args()

    metrics = _load_json(args.primary_dir / "metrics.json")
    rows = list(metrics.get("rollouts", []))
    if args.continuation_dir is not None:
        continuation_metrics = _load_json(args.continuation_dir / "metrics.json")
        rows.extend(continuation_metrics.get("rollouts", []))
    rows.sort(key=lambda row: int(row["rollout_index"]))
    if not rows:
        raise RuntimeError("no completed rollout metrics found")

    maximum = max(int(row["rollout_index"]) for row in rows)
    windows = [(1, 50), (151, 200)]
    if maximum > 200:
        windows.extend((start, min(start + 199, maximum)) for start in range(201, maximum + 1, 200))
        windows.append((max(1, maximum - 99), maximum))
    trend = {
        "completed_rollouts": len(rows),
        "maximum_rollout_index": maximum,
        "windows": [aggregate_window(rows, start, end) for start, end in windows],
    }

    output_dir = args.output_dir or args.continuation_dir or args.primary_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trend_report.json").write_text(
        json.dumps(trend, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    primary_summary = _load_json(args.primary_dir / "nonfinite_summary.json")
    first = aggregate_window(rows, 1, 50)
    last = aggregate_window(rows, 151, 200)
    gate = evaluate_200_gate(metrics, primary_summary, first, last)
    (output_dir / "gate_report.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"TEACHER_DIAGNOSTIC_GATE={gate['status']}")
    return 0 if gate["status"] == "PASS" or not args.require_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
