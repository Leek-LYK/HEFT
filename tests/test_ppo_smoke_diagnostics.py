import unittest

import torch

from active_adaptation.utils.ppo_smoke import (
    gradient_report,
    parameter_delta,
    state_key_report,
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


if __name__ == "__main__":
    unittest.main()
