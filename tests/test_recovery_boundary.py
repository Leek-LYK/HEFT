"""Regression for a global forward corrupting another env during recovery."""
import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from active_adaptation.envs.base import _Env
from scripts.utils.helpers import MaskedVecNorm


class RecoveryBoundaryTest(unittest.TestCase):
    def test_recovery_rechecks_other_worlds_before_sensor_history(self):
        data = SimpleNamespace(
            qpos=torch.ones(3, 7), qvel=torch.ones(3, 6),
            qacc=torch.zeros(3, 6), qacc_warmstart=torch.zeros(3, 6),
            sensordata=torch.zeros(3, 4),
        )
        before = data.qpos.clone()
        reset_rows = []
        calls = []

        def forward():
            calls.append(1)
            if len(calls) == 1:
                # First recovery recomputes all worlds: previously healthy
                # env 1 becomes nonfinite although only env 0 was reset.
                data.sensordata[1] = float("nan")
            else:
                # Repeated forward also recreates scratch NaNs in the env
                # recovered in the previous pass; the quarantine is cumulative.
                data.qacc[0] = float("nan")
                data.sensordata[0] = float("nan")

        env = SimpleNamespace(
            sim=SimpleNamespace(data=data, reset=lambda ids: None,
                                forward=forward, sense=lambda: None),
            scene=SimpleNamespace(reset=lambda ids: reset_rows.extend(ids.tolist())),
            _nonfinite_state_mask=torch.tensor([True, False, False]),
            nonfinite_state_recovery_count=1, nonfinite_diagnostics=None,
            input_tensordict=None, timestamp=42,
            _nonfinite_diagnostic_tensors=lambda td: vars(data),
            _all_finite_by_env=lambda fields: torch.isfinite(
                torch.cat(list(fields.values()), dim=1)
            ).all(1),
        )
        _Env._recover_nonfinite_envs(env, torch.tensor([0]), before)
        self.assertEqual(reset_rows, [0, 1])
        self.assertEqual(env._nonfinite_state_mask.tolist(), [True, True, False])
        self.assertEqual(env.nonfinite_state_recovery_count, 2)
        self.assertTrue(torch.isfinite(data.sensordata).all())
        torch.testing.assert_close(data.qpos, before)
        torch.testing.assert_close(data.qvel[2], torch.ones(6))

    def test_unmarked_nan_is_rejected_before_vecnorm_changes(self):
        norm = MaskedVecNorm(["obs"], decay=1.0)
        norm(TensorDict({"obs": torch.ones(2, 1)}, [2]))
        before = norm._td.clone()
        with self.assertRaisesRegex(FloatingPointError, "input to VecNorm"):
            norm(TensorDict({"obs": torch.tensor([[2.0], [float("nan")]])}, [2]))
        for key in before.keys():
            torch.testing.assert_close(before[key], norm._td[key])

    def test_marked_nan_is_excluded_and_returns_finite(self):
        norm = MaskedVecNorm(["obs"], decay=1.0)
        td = TensorDict({"obs": torch.tensor([[2.0], [float("nan")]]),
                         "_vecnorm_valid": torch.tensor([[True], [False]])}, [2])
        result = norm(td)
        self.assertTrue(torch.isfinite(result["obs"]).all())
        self.assertEqual(norm.loc["obs"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
