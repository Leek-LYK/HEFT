import copy
import unittest
import warnings
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.modules.student_latent_quantizer import (
    DEFAULT_STUDENT_LATENT_QUANTIZER_SCALE,
    StudentLatentQuantizer,
    StudentLatentQuantizerCfg,
)
from active_adaptation.learning.ppo.ppo import (
    PPOConfig,
    PPOPolicy,
    student_estimator_mse,
)


class _ActionManager:
    joint_names = [f"joint_{i}" for i in range(21)]


class StudentLatentQuantizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = SimpleNamespace(
            action_manager=_ActionManager(),
            cfg=SimpleNamespace(reward={}),
        )
        cls.observation_spec = Composite(
            policy=Unbounded(shape=(1, 1313)),
            priv=Unbounded(shape=(1, 64)),
            priv_critic=Unbounded(shape=(1, 8)),
            shape=(1,),
        )
        action_spec = Unbounded(shape=(1, 21))
        cfg = PPOConfig(
            phase="adapt",
            latent_dim=40,
            optimizer="adam",
            symmetry_enabled=False,
            student_latent_quantizer=StudentLatentQuantizerCfg(
                enabled=True,
                scale=4.499549865722656,
            ),
        )
        cls.policy = PPOPolicy(
            cfg,
            cls.observation_spec,
            action_spec,
            reward_spec=None,
            device="cpu",
            env=env,
        )

    def test_disabled_quantizer_is_the_original_continuous_path(self):
        latent = torch.randn(3, 40, requires_grad=True)
        output = StudentLatentQuantizer(enabled=False)(latent)
        self.assertIs(output, latent)
        torch.testing.assert_close(output, latent)

    def test_enabled_quantizer_levels_shape_scale_and_ste_gradient(self):
        latent = torch.linspace(-2.0, 2.0, 120).reshape(3, 40).requires_grad_()
        quantizer = StudentLatentQuantizer(enabled=True)
        output = quantizer(latent)

        self.assertEqual(output.shape, latent.shape)
        self.assertEqual(quantizer.scale, DEFAULT_STUDENT_LATENT_QUANTIZER_SCALE)
        self.assertEqual(quantizer.scale, 4.499549865722656)
        self.assertGreaterEqual(output.min(), -4.0)
        self.assertLessEqual(output.max(), 4.0)
        torch.testing.assert_close(output, output.round())

        output.sum().backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(torch.count_nonzero(latent.grad), 0)

    def test_estimator_loss_uses_continuous_student_and_teacher_latents(self):
        priv_pred = torch.tensor([[0.2, 0.7], [-0.4, 0.3]], requires_grad=True)
        priv_feature = torch.tensor([[0.0, 1.0], [0.5, -0.5]])
        valid = torch.ones(2, 1, dtype=torch.bool)
        quantized = StudentLatentQuantizer(enabled=True)(priv_pred)

        loss = student_estimator_mse(priv_pred, priv_feature, valid)
        torch.testing.assert_close(loss, F.mse_loss(priv_pred, priv_feature))
        self.assertFalse(torch.isclose(loss, F.mse_loss(quantized, priv_feature)))
        loss.backward()
        self.assertIsNotNone(priv_pred.grad)

    def test_student_actor_consumes_quantized_feature_and_contract_dimensions(self):
        original_phase = self.policy.cfg.phase
        try:
            for phase in ("adapt", "finetune"):
                self.policy.cfg.phase = phase
                td = self.observation_spec.zero()
                self.policy.get_rollout_policy("deploy")(td)

                self.assertEqual(
                    self.policy.student_feature_key, "student_priv_feature"
                )
                self.assertEqual(td["priv_pred"].shape, (1, 40))
                self.assertEqual(td["student_priv_feature"].shape, (1, 40))
                torch.testing.assert_close(
                    td["student_priv_feature"], td["student_priv_feature"].round()
                )
                self.assertEqual(td["_actor_inp"].shape, (1, 1353))
                torch.testing.assert_close(
                    td["_actor_inp"][:, -40:], td["student_priv_feature"]
                )
                self.assertEqual(td["action"].shape, (1, 21))
        finally:
            self.policy.cfg.phase = original_phase

    def test_teacher_actor_keeps_continuous_feature_and_quantizer_has_no_parameters(self):
        original_phase = self.policy.cfg.phase
        try:
            self.policy.cfg.phase = "train"
            td = self.observation_spec.zero()
            self.policy.get_rollout_policy("train")(td)
        finally:
            self.policy.cfg.phase = original_phase

        self.assertNotIn("student_priv_feature", td.keys())
        self.assertEqual(self.policy.teacher_feature_key, "priv_feature")
        torch.testing.assert_close(td["_actor_inp"][:, -40:], td["priv_feature"])
        self.assertEqual(list(self.policy.student_latent_quantizer.parameters()), [])
        estimator_params = {
            id(parameter)
            for group in self.policy.opt_estimator.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(
            estimator_params,
            {id(parameter) for parameter in self.policy.adapt_module.parameters()},
        )

    def test_official_default_config_remains_unquantized_256_dimensional(self):
        cfg = PPOConfig()
        self.assertEqual(cfg.latent_dim, 256)
        self.assertFalse(cfg.student_latent_quantizer.enabled)

    def test_checkpoint_without_quantizer_state_remains_loadable(self):
        old_state = self.policy.state_dict()
        old_state.pop("student_latent_quantizer")
        old_state["last_phase"] = "adapt"
        self.policy.load_state_dict(old_state, strict=True)

    def test_train_checkpoint_hard_copy_keeps_40_dimensional_actors_compatible(self):
        original_state = copy.deepcopy(self.policy.state_dict())
        state = copy.deepcopy(original_state)
        self.assertEqual(
            set(state["actor_teacher"]), set(state["actor_student"])
        )
        for value in state["actor_teacher"].values():
            value.fill_(0.125)
        for value in state["actor_student"].values():
            value.fill_(-0.25)
        state["last_phase"] = "train"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.policy.load_state_dict(state, strict=True)
            teacher_state = self.policy.actor_teacher.state_dict()
            student_state = self.policy.actor_student.state_dict()
            for key in teacher_state:
                torch.testing.assert_close(teacher_state[key], student_state[key])
        finally:
            original_state["last_phase"] = "adapt"
            self.policy.load_state_dict(original_state, strict=True)

    def test_student_ppo_forward_runs_the_quantizer(self):
        original_state = copy.deepcopy(self.policy.state_dict())
        td = self.observation_spec.expand(2).zero()
        with torch.no_grad():
            self.policy.get_rollout_policy("train")(td)
        td["is_init"] = torch.zeros(2, 1, dtype=torch.bool)
        td["adv"] = torch.ones(2, 1)
        td["ret"] = torch.zeros(2, 1)

        calls = 0

        def count_call(_module, _args, _output):
            nonlocal calls
            calls += 1

        hook = self.policy.student_latent_quantizer.register_forward_hook(count_call)
        try:
            self.policy._update_ppo_batch(td, mode="student")
            self.assertEqual(calls, 1)
            self.assertTrue(
                all(parameter.grad is None for parameter in self.policy.adapt_module.parameters())
            )
        finally:
            hook.remove()
            original_state["last_phase"] = "adapt"
            self.policy.load_state_dict(original_state, strict=True)


if __name__ == "__main__":
    unittest.main()
