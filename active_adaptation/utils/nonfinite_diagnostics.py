"""Bounded event recording for simulator non-finite recovery.

The recorder deliberately knows nothing about policies or simulator tensors.  The
environment reduces those tensors to scalar event payloads and this module owns
bounded JSONL output, aggregate counters, and abort-threshold decisions.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


NONFINITE_FIELD_ORDER = (
    "qpos",
    "qvel",
    "qacc",
    "qacc_warmstart",
    "ctrl",
    "sensordata",
    "action",
    "joint_target",
    "actuator_force",
    "contact_force",
)
PRE_STEP_ABORT_FIELDS = frozenset(("action", "joint_target", "ctrl"))
CORE_STATE_FIELDS = frozenset(("qpos", "qvel", "qacc"))


class NonfiniteAbortError(RuntimeError):
    """Raised after an event trips a configured diagnostic abort threshold."""


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _json_safe(value: Any) -> Any:
    """Return strict-JSON data while retaining non-finite scalar information."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (RuntimeError, ValueError):
            pass
    return str(value)


class NonfiniteEventRecorder:
    """Write bounded event details and unbounded aggregate counts.

    Event rates use environment-level denominators: one vectorized policy step
    contributes ``num_envs`` policy transitions and each executed simulator step
    contributes ``num_envs`` physics substeps.
    """

    SUMMARY_VERSION = 1

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        num_envs: int,
        decimation: int,
    ) -> NonfiniteEventRecorder | None:
        if not bool(_cfg_get(cfg, "enabled", False)):
            return None
        return cls(cfg, num_envs=num_envs, decimation=decimation)

    def __init__(self, cfg: Any, *, num_envs: int, decimation: int) -> None:
        self.num_envs = int(num_envs)
        self.decimation = int(decimation)
        if self.num_envs <= 0 or self.decimation <= 0:
            raise ValueError("num_envs and decimation must be positive")

        raw_output_dir = _cfg_get(cfg, "output_dir", None)
        if not raw_output_dir:
            raise ValueError("enabled nonfinite diagnostics require output_dir")
        self.output_dir = Path(str(raw_output_dir)).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "nonfinite_events.jsonl"
        self.summary_path = self.output_dir / "nonfinite_summary.json"

        self.detailed_event_limit = int(_cfg_get(cfg, "detailed_event_limit", 200))
        self.rate_min_policy_transitions = int(
            _cfg_get(cfg, "rate_min_policy_transitions", 100_000)
        )
        self.max_event_rate = float(_cfg_get(cfg, "max_event_rate", 1.0e-4))
        self.max_motion_frame_repeats = int(
            _cfg_get(cfg, "max_motion_frame_repeats", 10)
        )
        self.max_events_per_iteration = int(
            _cfg_get(cfg, "max_events_per_iteration", 20)
        )
        if self.detailed_event_limit < 0:
            raise ValueError("detailed_event_limit must be non-negative")
        if self.rate_min_policy_transitions < 0 or self.max_event_rate < 0.0:
            raise ValueError("nonfinite event-rate thresholds must be non-negative")
        if self.max_motion_frame_repeats < 0 or self.max_events_per_iteration < 0:
            raise ValueError("nonfinite count thresholds must be non-negative")

        self.total_events = 0
        self.policy_transitions = 0
        self.physics_substeps = 0
        self.detailed_events_written = 0
        self.events_by_first_field: Counter[str] = Counter()
        self.events_by_nonfinite_field: Counter[str] = Counter()
        self.events_by_physics_substep: Counter[str] = Counter()
        self.events_by_motion: Counter[str] = Counter()
        self.events_by_motion_frame: Counter[str] = Counter()
        self.events_by_reference_frame: Counter[str] = Counter()
        self.events_by_iteration: Counter[str] = Counter()
        self.pre_step_nonfinite_events = 0
        self.post_step_new_nonfinite_events = 0
        self.only_qacc_warmstart_events = 0
        self.core_state_nonfinite_events = 0
        self.pre_step_control_nonfinite_events = 0
        self.pre_step_control_nonfinite_by_field: Counter[str] = Counter()
        self.event_env_ids: set[int] = set()
        self.motion_context: dict[str, dict[str, Any]] = {}
        self.max_contiguous_event_burst = 0
        self._current_contiguous_event_burst = 0
        self._last_event_physics_tick: int | None = None
        self.abort_reason: dict[str, Any] | None = None
        self.status = "RUNNING"

        self._load_initial_state(
            _cfg_get(cfg, "initial_summary_path", None),
            _cfg_get(cfg, "initial_events_path", None),
        )
        self.flush_summary()

    def _load_initial_state(
        self,
        initial_summary_path: str | None,
        initial_events_path: str | None,
    ) -> None:
        initial_summary = None
        if initial_summary_path:
            path = Path(str(initial_summary_path)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"initial nonfinite summary not found: {path}")
            initial_summary = json.loads(path.read_text(encoding="utf-8"))
            self.total_events = int(initial_summary.get("total_events", 0))
            self.policy_transitions = int(
                initial_summary.get("policy_transitions", 0)
            )
            self.physics_substeps = int(initial_summary.get("physics_substeps", 0))
            self.events_by_first_field.update(
                initial_summary.get("events_by_first_nonfinite_field", {})
            )
            self.events_by_nonfinite_field.update(
                initial_summary.get("events_by_nonfinite_field", {})
            )
            self.events_by_physics_substep.update(
                initial_summary.get("events_by_physics_substep", {})
            )
            self.events_by_motion.update(initial_summary.get("events_by_motion", {}))
            self.events_by_motion_frame.update(
                initial_summary.get("events_by_motion_frame", {})
            )
            self.events_by_reference_frame.update(
                initial_summary.get("events_by_reference_motion_frame", {})
            )
            self.events_by_iteration.update(
                initial_summary.get("events_by_iteration", {})
            )
            self.pre_step_nonfinite_events = int(
                initial_summary.get("pre_step_nonfinite_events", 0)
            )
            self.post_step_new_nonfinite_events = int(
                initial_summary.get("post_step_new_nonfinite_events", 0)
            )
            self.only_qacc_warmstart_events = int(
                initial_summary.get("only_qacc_warmstart_events", 0)
            )
            self.core_state_nonfinite_events = int(
                initial_summary.get("core_state_nonfinite_events", 0)
            )
            self.pre_step_control_nonfinite_events = int(
                initial_summary.get("pre_step_control_nonfinite_events", 0)
            )
            self.pre_step_control_nonfinite_by_field.update(
                initial_summary.get("pre_step_control_nonfinite_by_field", {})
            )
            self.event_env_ids.update(
                int(value) for value in initial_summary.get("event_env_ids", [])
            )
            self.motion_context.update(initial_summary.get("motion_context", {}))
            self.max_contiguous_event_burst = int(
                initial_summary.get("max_contiguous_event_burst", 0)
            )
            continuation = initial_summary.get("continuation_state", {})
            last_tick = continuation.get("last_event_physics_tick")
            self._last_event_physics_tick = (
                None if last_tick is None else int(last_tick)
            )
            self._current_contiguous_event_burst = int(
                continuation.get("current_contiguous_event_burst", 0)
            )

        # Always create/truncate the destination.  If this is a continuation,
        # retain only the already-bounded detailed prefix.
        self.events_path.write_text("", encoding="utf-8")
        if initial_events_path:
            path = Path(str(initial_events_path)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"initial nonfinite events not found: {path}")
            copied = 0
            with path.open("r", encoding="utf-8") as source, self.events_path.open(
                "a", encoding="utf-8"
            ) as destination:
                for line in source:
                    if copied >= self.detailed_event_limit:
                        break
                    if line.strip():
                        destination.write(line.rstrip("\n") + "\n")
                        copied += 1
            self.detailed_events_written = copied
        elif initial_summary is not None:
            self.detailed_events_written = min(
                int(initial_summary.get("detailed_events_written", 0)),
                self.detailed_event_limit,
            )

    def note_policy_transition_batch(self) -> None:
        self.policy_transitions += self.num_envs

    def note_physics_substep_batch(self) -> None:
        self.physics_substeps += self.num_envs

    @staticmethod
    def _counter_dict(counter: Counter[str]) -> dict[str, int]:
        return {str(key): int(counter[key]) for key in sorted(counter)}

    def _rate_abort(self) -> dict[str, Any] | None:
        if self.policy_transitions < self.rate_min_policy_transitions:
            return None
        rate = self.total_events / max(self.policy_transitions, 1)
        if rate > self.max_event_rate:
            return {
                "code": "event_rate_exceeded",
                "event_rate": rate,
                "threshold": self.max_event_rate,
                "policy_transitions": self.policy_transitions,
                "events": self.total_events,
            }
        return None

    def _set_abort(self, reason: dict[str, Any] | None) -> None:
        if reason is not None and self.abort_reason is None:
            self.abort_reason = _json_safe(reason)
            self.status = "ABORT_REQUESTED"

    def _update_burst(self, physics_tick: int) -> None:
        if self._last_event_physics_tick is None:
            self._current_contiguous_event_burst = 1
        elif physics_tick in {
            self._last_event_physics_tick,
            self._last_event_physics_tick + 1,
        }:
            self._current_contiguous_event_burst += 1
        else:
            self._current_contiguous_event_burst = 1
        self._last_event_physics_tick = physics_tick
        self.max_contiguous_event_burst = max(
            self.max_contiguous_event_burst,
            self._current_contiguous_event_burst,
        )

    def record_events(self, events: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
        for raw_event in events:
            event = _json_safe(dict(raw_event))
            self.total_events += 1
            first_field = str(event["first_nonfinite_field"])
            self.events_by_first_field[first_field] += 1
            for field in event.get("nonfinite_fields", []):
                self.events_by_nonfinite_field[str(field)] += 1

            substep = str(event["physics_substep"])
            iteration = str(event["policy_iteration"])
            motion_id = str(event["motion_id"])
            reference_frame = str(event["reference_motion_frame"])
            motion_frame_key = f"{motion_id}:{reference_frame}"
            self.events_by_physics_substep[substep] += 1
            self.events_by_iteration[iteration] += 1
            self.events_by_motion[motion_id] += 1
            self.events_by_motion_frame[motion_frame_key] += 1
            self.events_by_reference_frame[reference_frame] += 1
            self.event_env_ids.add(int(event["env_id"]))
            self.motion_context[motion_id] = {
                "motion_label": event.get("motion_label"),
                "motion_relative_path": event.get("motion_relative_path"),
            }

            if event.get("classification") == "pre_step_nonfinite":
                self.pre_step_nonfinite_events += 1
            else:
                self.post_step_new_nonfinite_events += 1
            if bool(event.get("only_qacc_warmstart_nonfinite", False)):
                self.only_qacc_warmstart_events += 1
            if bool(event.get("core_state_nonfinite", False)):
                self.core_state_nonfinite_events += 1

            pre_control_fields = sorted(
                PRE_STEP_ABORT_FIELDS.intersection(
                    str(field) for field in event.get("pre_nonfinite_fields", [])
                )
            )
            if pre_control_fields:
                self.pre_step_control_nonfinite_events += 1
                self.pre_step_control_nonfinite_by_field.update(pre_control_fields)
                self._set_abort(
                    {
                        "code": "pre_step_control_nonfinite",
                        "fields": pre_control_fields,
                        "policy_iteration": int(event["policy_iteration"]),
                        "env_id": int(event["env_id"]),
                    }
                )

            self._update_burst(int(event["global_physics_tick"]))
            if self.detailed_events_written < self.detailed_event_limit:
                with self.events_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                    )
                self.detailed_events_written += 1

            repeats = self.events_by_motion_frame[motion_frame_key]
            if repeats > self.max_motion_frame_repeats:
                self._set_abort(
                    {
                        "code": "motion_frame_repeat_exceeded",
                        "motion_id": int(event["motion_id"]),
                        "reference_motion_frame": int(
                            event["reference_motion_frame"]
                        ),
                        "count": repeats,
                        "threshold": self.max_motion_frame_repeats,
                    }
                )
            iteration_events = self.events_by_iteration[iteration]
            if iteration_events > self.max_events_per_iteration:
                self._set_abort(
                    {
                        "code": "iteration_event_burst_exceeded",
                        "policy_iteration": int(event["policy_iteration"]),
                        "count": iteration_events,
                        "threshold": self.max_events_per_iteration,
                    }
                )

        self._set_abort(self._rate_abort())
        self.flush_summary()
        return self.abort_reason

    def raise_if_aborted(self) -> None:
        if self.abort_reason is not None:
            raise NonfiniteAbortError(
                "nonfinite diagnostic abort: "
                + json.dumps(self.abort_reason, sort_keys=True, allow_nan=False)
            )

    def most_repeated_motion_frames(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = []
        for key, count in self.events_by_motion_frame.most_common(limit):
            motion_id, frame = key.split(":", 1)
            context = self.motion_context.get(motion_id, {})
            rows.append(
                {
                    "motion_id": int(motion_id),
                    "reference_motion_frame": int(frame),
                    "count": int(count),
                    **context,
                }
            )
        return rows

    def summary(self) -> dict[str, Any]:
        policy_rate = self.total_events / max(self.policy_transitions, 1)
        physics_rate = self.total_events / max(self.physics_substeps, 1)
        return {
            "version": self.SUMMARY_VERSION,
            "status": self.status,
            "abort_reason": self.abort_reason,
            "total_events": int(self.total_events),
            "policy_transitions": int(self.policy_transitions),
            "physics_substeps": int(self.physics_substeps),
            "event_rate_per_policy_transition": policy_rate,
            "event_rate_per_physics_substep": physics_rate,
            "events_per_million_policy_transitions": policy_rate * 1.0e6,
            "events_per_million_physics_substeps": physics_rate * 1.0e6,
            "events_by_first_nonfinite_field": self._counter_dict(
                self.events_by_first_field
            ),
            "events_by_nonfinite_field": self._counter_dict(
                self.events_by_nonfinite_field
            ),
            "events_by_physics_substep": self._counter_dict(
                self.events_by_physics_substep
            ),
            "events_by_motion": self._counter_dict(self.events_by_motion),
            "events_by_motion_frame": self._counter_dict(
                self.events_by_motion_frame
            ),
            "events_by_reference_motion_frame": self._counter_dict(
                self.events_by_reference_frame
            ),
            "events_by_iteration": self._counter_dict(self.events_by_iteration),
            "most_repeated_motion_frames": self.most_repeated_motion_frames(),
            "max_motion_frame_repeat": max(
                self.events_by_motion_frame.values(), default=0
            ),
            "max_events_in_iteration": max(
                self.events_by_iteration.values(), default=0
            ),
            "max_contiguous_event_burst": int(
                self.max_contiguous_event_burst
            ),
            "independent_env_count": len(self.event_env_ids),
            "event_env_ids": sorted(self.event_env_ids),
            "pre_step_nonfinite_events": int(self.pre_step_nonfinite_events),
            "post_step_new_nonfinite_events": int(
                self.post_step_new_nonfinite_events
            ),
            "only_qacc_warmstart_events": int(
                self.only_qacc_warmstart_events
            ),
            "core_state_nonfinite_events": int(
                self.core_state_nonfinite_events
            ),
            "pre_step_control_nonfinite_events": int(
                self.pre_step_control_nonfinite_events
            ),
            "pre_step_control_nonfinite_by_field": self._counter_dict(
                self.pre_step_control_nonfinite_by_field
            ),
            "detailed_event_limit": int(self.detailed_event_limit),
            "detailed_events_written": int(self.detailed_events_written),
            "motion_context": _json_safe(self.motion_context),
            "thresholds": {
                "rate_min_policy_transitions": self.rate_min_policy_transitions,
                "max_event_rate": self.max_event_rate,
                "max_motion_frame_repeats": self.max_motion_frame_repeats,
                "max_events_per_iteration": self.max_events_per_iteration,
            },
            "continuation_state": {
                "last_event_physics_tick": self._last_event_physics_tick,
                "current_contiguous_event_burst": (
                    self._current_contiguous_event_burst
                ),
            },
        }

    def flush_summary(self, *, status: str | None = None) -> dict[str, Any]:
        if status is not None:
            self.status = str(status)
        summary = _json_safe(self.summary())
        temporary_path = self.summary_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(temporary_path), str(self.summary_path))
        return summary
