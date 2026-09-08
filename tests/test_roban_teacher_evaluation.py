from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_roban_teacher import (
    EpisodeAccumulator,
    _parameter_delta,
    _parameter_snapshot,
    _tensor_state_delta,
    _tensor_state_snapshot,
    summarize_root_errors,
    summarize_samples,
)


def test_sample_summary_reports_mean_and_p95():
    values = torch.arange(1, 101, dtype=torch.float32)
    summary = summarize_samples(values)
    assert summary["count"] == 100
    assert summary["mean"] == pytest.approx(50.5)
    assert summary["p95"] == pytest.approx(95.05)


def test_empty_root_error_summary_is_explicit():
    empty = torch.empty(0)
    summary = summarize_root_errors(empty, empty, empty)
    assert summary["root_xy_position_error_m"]["count"] == 0
    assert summary["root_z_error_m"]["mean"] is None


def test_episode_summary_has_per_episode_and_per_step_components():
    accumulator = EpisodeAccumulator(torch.device("cpu"))
    accumulator.completed.fill_(2)
    accumulator.terminated.fill_(1)
    accumulator.immediate.fill_(1)
    accumulator.stat_sums = {
        "episode_len": torch.tensor(20.0, dtype=torch.float64),
        "tracking/return": torch.tensor(8.0, dtype=torch.float64),
        "loco/return": torch.tensor(2.0, dtype=torch.float64),
        "tracking/root_pos_tracking": torch.tensor(4.0, dtype=torch.float64),
        "termination/body_z_termination": torch.tensor(1.0, dtype=torch.float64),
    }
    summary = accumulator.summary()
    assert summary["mean_episode_reward"] == pytest.approx(5.0)
    assert summary["mean_episode_length"] == pytest.approx(10.0)
    assert summary["body_z_termination_ratio"] == pytest.approx(0.5)
    assert summary["termination_le_10_steps_ratio"] == pytest.approx(0.5)
    component = summary["reward_components"]["tracking/root_pos_tracking"]
    assert component["per_episode_cumulative"] == pytest.approx(2.0)
    assert component["per_step"] == pytest.approx(0.2)


def test_policy_and_vecnorm_snapshots_are_exact_when_eval_is_read_only():
    policy = torch.nn.Linear(3, 2)
    policy_before = _parameter_snapshot(policy)
    vecnorm_state = {
        "policy_sum": torch.tensor([1.0, 2.0]),
        "policy_count": torch.tensor(3.0),
    }
    vecnorm_before = _tensor_state_snapshot(vecnorm_state)
    assert _parameter_delta(policy_before, policy) == {
        "changed": False,
        "changed_parameter_count": 0,
        "changed_parameters": [],
        "max_abs_delta": 0.0,
    }
    assert _tensor_state_delta(vecnorm_before, vecnorm_state) == {
        "changed": False,
        "changed_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "max_abs_delta": 0.0,
    }
