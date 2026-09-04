import json
import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.envs.base import apply_termination_grace_and_recovery
from active_adaptation.utils.nonfinite_diagnostics import (
    NonfiniteAbortError,
    NonfiniteEventRecorder,
)

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.utils.helpers import MaskedVecNorm


def _event(
    *,
    iteration=1,
    env_id=0,
    motion_id=2,
    frame=30,
    tick=10,
    substep=0,
    first_field="qacc_warmstart",
    pre_bad=(),
):
    nonfinite = list(pre_bad) or [first_field]
    return {
        "policy_iteration": iteration,
        "global_policy_frame": 256,
        "env_id": env_id,
        "physics_substep": substep,
        "global_physics_tick": tick,
        "episode_length": 7,
        "motion_id": motion_id,
        "motion_label": {"name": "test"},
        "motion_relative_path": "motion/test.npz",
        "reference_motion_frame": frame,
        "command_time_s": 0.6,
        "first_nonfinite_field": first_field,
        "first_nonfinite_flat_index": 3,
        "first_field_finite_before_sim_step": not pre_bad,
        "first_field_finite_after_sim_step": False,
        "pre_nonfinite_fields": list(pre_bad),
        "post_nonfinite_fields": ([] if pre_bad else [first_field]),
        "nonfinite_fields": nonfinite,
        "classification": (
            "pre_step_nonfinite" if pre_bad else "post_step_new_nonfinite"
        ),
        "sim_step_executed": not pre_bad,
        "only_qacc_warmstart_nonfinite": nonfinite == ["qacc_warmstart"],
        "core_state_nonfinite": bool({"qpos", "qvel", "qacc"} & set(nonfinite)),
        "qacc_max_abs": float("inf"),
    }


def _cfg(path, **overrides):
    cfg = {
        "enabled": True,
        "output_dir": str(path),
        "detailed_event_limit": 200,
        "rate_min_policy_transitions": 100_000,
        "max_event_rate": 1.0e-4,
        "max_motion_frame_repeats": 10,
        "max_events_per_iteration": 20,
    }
    cfg.update(overrides)
    return cfg


def test_default_off_does_not_create_event_files(tmp_path):
    recorder = NonfiniteEventRecorder.from_config(
        {"enabled": False, "output_dir": str(tmp_path / "off")},
        num_envs=2,
        decimation=4,
    )
    assert recorder is None
    assert not (tmp_path / "off").exists()


def test_event_json_is_strict_serializable_and_details_are_bounded(tmp_path):
    recorder = NonfiniteEventRecorder(
        _cfg(
            tmp_path,
            detailed_event_limit=2,
            max_event_rate=1.0,
            max_motion_frame_repeats=99,
            max_events_per_iteration=99,
        ),
        num_envs=2,
        decimation=4,
    )
    for _ in range(5):
        recorder.note_policy_transition_batch()
    for _ in range(4):
        recorder.note_physics_substep_batch()
    recorder.record_events(
        [
            _event(env_id=0, tick=10),
            _event(env_id=1, tick=10, first_field="qacc"),
            _event(env_id=0, tick=11, frame=31, first_field="qvel"),
        ]
    )

    lines = (tmp_path / "nonfinite_events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["qacc_max_abs"] == "inf"
    summary = json.loads((tmp_path / "nonfinite_summary.json").read_text())
    assert summary["total_events"] == 3
    assert summary["policy_transitions"] == 10
    assert summary["physics_substeps"] == 8
    assert summary["detailed_events_written"] == 2
    assert summary["events_by_first_nonfinite_field"] == {
        "qacc": 1,
        "qacc_warmstart": 1,
        "qvel": 1,
    }
    assert summary["max_contiguous_event_burst"] == 3
    assert summary["independent_env_count"] == 2


@pytest.mark.parametrize(
    ("overrides", "events", "transition_batches", "reason"),
    [
        (
            {"rate_min_policy_transitions": 2, "max_event_rate": 0.4},
            [_event()],
            2,
            "event_rate_exceeded",
        ),
        (
            {"max_motion_frame_repeats": 1, "max_event_rate": 1.0},
            [_event(), _event(env_id=1)],
            2,
            "motion_frame_repeat_exceeded",
        ),
        (
            {"max_events_per_iteration": 1, "max_event_rate": 1.0},
            [_event(), _event(env_id=1, frame=31)],
            2,
            "iteration_event_burst_exceeded",
        ),
        (
            {"max_event_rate": 1.0},
            [_event(first_field="action", pre_bad=("action",))],
            2,
            "pre_step_control_nonfinite",
        ),
    ],
)
def test_abort_rate_repeat_iteration_burst_and_pre_control(
    tmp_path, overrides, events, transition_batches, reason
):
    recorder = NonfiniteEventRecorder(
        _cfg(tmp_path, **overrides), num_envs=1, decimation=4
    )
    for _ in range(transition_batches):
        recorder.note_policy_transition_batch()
    recorder.record_events(events)
    assert recorder.abort_reason["code"] == reason
    with pytest.raises(NonfiniteAbortError):
        recorder.raise_if_aborted()


def test_continuation_preserves_aggregate_counts_and_detail_cap(tmp_path):
    first_dir = tmp_path / "first"
    first = NonfiniteEventRecorder(
        _cfg(
            first_dir,
            detailed_event_limit=2,
            max_event_rate=1.0,
            max_motion_frame_repeats=99,
        ),
        num_envs=1,
        decimation=4,
    )
    first.note_policy_transition_batch()
    first.note_physics_substep_batch()
    first.record_events([_event()])

    second_dir = tmp_path / "second"
    second = NonfiniteEventRecorder(
        _cfg(
            second_dir,
            detailed_event_limit=2,
            max_event_rate=1.0,
            max_motion_frame_repeats=99,
            initial_summary_path=str(first.summary_path),
            initial_events_path=str(first.events_path),
        ),
        num_envs=1,
        decimation=4,
    )
    second.note_policy_transition_batch()
    second.note_physics_substep_batch()
    second.record_events([_event(frame=31, tick=11)])
    summary = second.summary()
    assert summary["total_events"] == 2
    assert summary["policy_transitions"] == 2
    assert summary["physics_substeps"] == 2
    assert summary["detailed_events_written"] == 2
    assert len(second.events_path.read_text().splitlines()) == 2


def test_recovery_is_immediate_terminal_and_normal_rows_are_unchanged():
    terminated = torch.tensor([[False], [True], [True]])
    episode_length = torch.tensor([1, 10, 1])
    recovery = torch.tensor([True, False, False])
    result = apply_termination_grace_and_recovery(
        terminated, episode_length, recovery
    )
    assert result.squeeze(1).tolist() == [True, True, False]


def test_vecnorm_excludes_only_recovered_rows_from_running_moments():
    masked = MaskedVecNorm(["obs"], decay=1.0)
    td = TensorDict(
        {
            "obs": torch.tensor([[1.0], [100.0]]),
            "_vecnorm_valid": torch.tensor([[True], [False]]),
        },
        batch_size=[2],
    )
    masked(td)
    assert masked.loc["obs"].item() == pytest.approx(1.0)

    normal = MaskedVecNorm(["obs"], decay=1.0)
    normal(TensorDict({"obs": torch.tensor([[1.0], [3.0]])}, batch_size=[2]))
    assert normal.loc["obs"].item() == pytest.approx(2.0)
