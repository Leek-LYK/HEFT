import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tensordict import TensorDict

import active_adaptation.learning  # noqa: F401 - registers structured algo configs
from active_adaptation.assets.roban import ROBAN_JOINT_ORDER
from active_adaptation.envs.mdp.action import JointPosition
from active_adaptation.utils.helpers import TensorRingBuffer


CFG_DIR = Path(__file__).parents[1] / "cfg"


def _compose_roban():
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
            cfg = compose(config_name="train", overrides=["task/profile=Roban_tracking"])
        OmegaConf.resolve(cfg)
        return cfg
    finally:
        GlobalHydra.instance().clear()


class RobanPolicyContractTest(unittest.TestCase):
    def test_roban_1313_observation_and_40_dimensional_actor_contract(self):
        cfg = _compose_roban()
        policy_obs = cfg.task.observation.policy

        self.assertEqual(
            list(policy_obs.keys()),
            [
                "boot_indicator_state_obs",
                "command_obs",
                "target_joint_pos_obs",
                "target_root_z_obs",
                "target_projected_gravity_b_obs",
                "root_angvel_b_history",
                "projected_gravity_history",
                "joint_pos_history",
                "joint_vel_history",
                "prev_actions",
            ],
        )

        joint_count = len(ROBAN_JOINT_ORDER)
        future_count = len(cfg.task.command.student_future_steps)
        history_steps = list(policy_obs.root_angvel_b_history.history_steps)
        previous_action_count = int(policy_obs.prev_actions.steps)

        self.assertEqual(joint_count, 21)
        self.assertEqual(future_count, 13)
        self.assertEqual(history_steps, [0, 1, 2, 3, 4, 8, 12, 16, 20])
        self.assertEqual(
            list(policy_obs.projected_gravity_history.history_steps), history_steps
        )
        self.assertEqual(list(policy_obs.joint_pos_history.history_steps), history_steps)
        self.assertEqual(list(policy_obs.joint_vel_history.history_steps), history_steps)
        self.assertEqual(previous_action_count, 8)

        history_count = len(history_steps)
        observation_dim = (
            1
            + (future_count - 1) * 3
            + future_count * 6
            + future_count * joint_count
            + future_count * joint_count
            + future_count
            + future_count * 3
            + history_count * 3
            + history_count * 3
            + history_count * joint_count
            + history_count * joint_count
            + previous_action_count * joint_count
        )
        latent_dim = int(cfg.algo.latent_dim)

        self.assertEqual(observation_dim, 1313)
        self.assertEqual(latent_dim, 40)
        self.assertEqual(observation_dim + latent_dim, 1353)
        self.assertEqual(joint_count, 21)

    def test_training_and_deployment_action_contracts_are_distinct(self):
        cfg = _compose_roban()
        deployment = cfg.task.profile.deployment

        self.assertFalse(deployment.residual_action)
        self.assertEqual(deployment.default_joint_position, 0.0)
        self.assertEqual(deployment.action_clip, 10.0)
        self.assertEqual(deployment.action_alpha, 0.9)
        self.assertEqual(list(cfg.task.action.alpha), [0.8, 1.0])
        self.assertEqual(cfg.task.action.prev_action_obs, "applied")
        self.assertNotIn("joint_direction", cfg.task)

    def test_applied_action_history_records_the_filtered_command(self):
        action = object.__new__(JointPosition)
        action.env = type("Env", (), {"decimation": 2})()
        action._action_buf = TensorRingBuffer(
            1, 8, 1, device="cpu", dtype=torch.float32
        )
        action._action_mean_buf = TensorRingBuffer(
            1, 8, 1, device="cpu", dtype=torch.float32
        )
        action._action_applied_buf = TensorRingBuffer(
            1, 8, 1, device="cpu", dtype=torch.float32
        )
        action.prev_action_obs = "applied"
        action.action_rate_source = "sampled"
        action.delay = torch.zeros(1, 1, dtype=torch.long)
        action.applied_action = torch.zeros(1, 1)
        action.alpha = torch.full((1, 1), 0.5)
        action.default_joint_pos = torch.zeros(1, 1)
        action.offset = torch.zeros(1, 1)
        action.joint_ids = [0]
        action.action_scaling = torch.ones(1)
        action.boot_protect_enabled = False

        class Asset:
            def set_joint_position_target(self, target):
                self.target = target.clone()

        action.asset = Asset()
        td = TensorDict(
            {"action": torch.ones(1, 1), "loc": torch.ones(1, 1)},
            batch_size=[1],
        )

        action(td, substep=0)
        torch.testing.assert_close(
            action.get_recent_action_obs(1), torch.zeros(1, 1, 1)
        )
        action(td, substep=1)
        torch.testing.assert_close(
            action.get_recent_action_obs(1), torch.full((1, 1, 1), 0.75)
        )


if __name__ == "__main__":
    unittest.main()
