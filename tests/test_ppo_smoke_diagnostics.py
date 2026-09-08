import math
import unittest

import torch

from active_adaptation.utils.ppo_smoke import (
    gradient_report,
    optimizer_finite_report,
    parameter_delta,
    root_tracking_error_summary,
    state_key_report,
    validate_pilot_schedule,
    validate_smoke_schedule,
)


class PPOSmokeDiagnosticsTest(unittest.TestCase):
    def test_schedule_is_no_drop_and_has_two_updates(self):
        schedule = validate_smoke_schedule(
            num_envs=64,
            rollout_horizon=8,
            num_minibatches=1,
            total_frames=1024,
        )
        self.assertEqual(schedule["batch_size"], 512)
        self.assertEqual(schedule["minibatch_size"], 512)
        self.assertEqual(schedule["estimator_sequences_per_minibatch"], 64)
        self.assertEqual(schedule["rollout_updates"], 2)

    def test_schedule_rejects_dropped_estimator_sequences(self):
        with self.assertRaisesRegex(ValueError, "num_envs must divide evenly"):
            validate_smoke_schedule(
                num_envs=65,
                rollout_horizon=8,
                num_minibatches=2,
                total_frames=1040,
            )

    def test_formal_pilot_schedule_keeps_production_rollout_and_minibatches(self):
        schedule = validate_pilot_schedule(
            num_envs=256,
            rollout_horizon=32,
            num_minibatches=8,
            total_frames=256 * 32 * 20,
            target_rollout_updates=20,
        )
        self.assertEqual(schedule["batch_size"], 8192)
        self.assertEqual(schedule["minibatch_size"], 1024)
        self.assertEqual(schedule["estimator_sequences_per_minibatch"], 32)
        self.assertEqual(schedule["rollout_updates"], 20)

    def test_formal_pilot_schedule_uses_global_frame_budget_for_ddp(self):
        schedule = validate_pilot_schedule(
            num_envs=2048,
            rollout_horizon=32,
            num_minibatches=8,
            total_frames=2048 * 32 * 100 * 3,
            target_rollout_updates=100,
            world_size=3,
        )
        self.assertEqual(schedule["batch_size"], 2048 * 32)
        self.assertEqual(schedule["global_batch_size"], 2048 * 32 * 3)
        self.assertEqual(schedule["rollout_updates"], 100)

    def test_optimizer_finite_report_detects_corruption(self):
        module = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(module.parameters())
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.assertTrue(optimizer_finite_report(optimizer)["finite"])
        next(iter(optimizer.state.values()))["exp_avg"].fill_(float("nan"))
        self.assertFalse(optimizer_finite_report(optimizer)["finite"])

    def test_parameter_gradient_and_state_reports(self):
        module = torch.nn.Linear(3, 2)
        before = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }
        module(torch.ones(4, 3)).sum().backward()
        grad = gradient_report(module)
        self.assertTrue(grad["finite"])
        self.assertEqual(grad["missing_gradients"], [])
        self.assertGreater(grad["gradient_norm"], 0.0)

        unchanged = parameter_delta(before, module)
        self.assertFalse(unchanged["changed"])
        with torch.no_grad():
            module.weight.add_(0.25)
        changed = parameter_delta(before, module)
        self.assertTrue(changed["changed"])
        self.assertGreater(changed["max_abs_delta"], 0.0)

        keys = state_key_report(module.state_dict(), module)
        self.assertEqual(keys, {"missing_keys": [], "unexpected_keys": []})

    def test_regular_module_child_is_not_mistaken_for_ddp(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.module = torch.nn.Linear(3, 2)

        wrapper = Wrapper()
        self.assertIn("module.weight", wrapper.state_dict())
        self.assertEqual(
            state_key_report(wrapper.state_dict(), wrapper),
            {"missing_keys": [], "unexpected_keys": []},
        )

    def test_root_tracking_error_summary_uses_xy_z_and_geodesic(self):
        current_pos = torch.tensor([[3.0, 4.0, 2.0]])
        target_pos = torch.tensor([[0.0, 0.0, 1.5]])
        current_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        target_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        report = root_tracking_error_summary(
            current_pos, target_pos, current_quat, target_quat
        )
        self.assertEqual(report["root_xy_position_error_m"]["mean"], 5.0)
        self.assertEqual(report["root_z_error_m"]["mean"], 0.5)
        self.assertAlmostEqual(
            report["root_orientation_geodesic_error_rad"]["mean"],
            math.pi,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
