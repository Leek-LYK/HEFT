import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import active_adaptation.learning  # noqa: F401 - registers structured algo configs
from active_adaptation.assets import G1_CFG, G1_CFG_5, XBOT_L7_CFG, get_robot_cfg
from active_adaptation.assets.roban import ROBAN_CFG, ROBAN_JOINT_ORDER
from active_adaptation.envs.locomotion import resolve_contact_body_pattern
from active_adaptation.envs.mdp.randomizations.core import perturb_body_wrench
from active_adaptation.envs.mdp.rewards.locomotion import (
    _feet_sides,
    feet_air_time_ref_dense,
)


CFG_DIR = Path(__file__).parents[1] / "cfg"


def _compose_profile(profile: str, extra_overrides: tuple[str, ...] = ()):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
            cfg = compose(
                config_name="train",
                overrides=[f"task/profile={profile}", *extra_overrides],
            )
        OmegaConf.resolve(cfg)
        return cfg
    finally:
        GlobalHydra.instance().clear()


def _matches(pattern: str, names) -> list[str]:
    return [name for name in names if re.fullmatch(pattern, name)]


class RobanProfileTest(unittest.TestCase):
    def test_roban_foot_names_resolve_left_and_right_sides(self):
        self.assertEqual(
            _feet_sides(["leg_l6_link", "leg_r6_link"], context="Roban feet"),
            ["left", "right"],
        )

    def test_registry_adds_roban_without_changing_existing_entries(self):
        self.assertIs(get_robot_cfg("roban"), ROBAN_CFG)
        self.assertIs(get_robot_cfg("g1"), G1_CFG)
        self.assertIs(get_robot_cfg("g1_5"), G1_CFG_5)
        self.assertIs(get_robot_cfg("xbot_l7"), XBOT_L7_CFG)
        with self.assertRaisesRegex(ValueError, "Unknown robot name"):
            get_robot_cfg("does_not_exist")

    def test_roban_profile_composes_and_body_patterns_exist(self):
        cfg = _compose_profile("Roban_tracking")
        model = ROBAN_CFG.spec_fn().compile()
        body_names = [model.body(i).name for i in range(model.nbody)]

        self.assertEqual(cfg.task.name, "ROBANTRACKING")
        self.assertEqual(cfg.task.robot.name, "roban")
        self.assertEqual(cfg.task.robot.mass, 40.69991336)
        for key in (
            "base_pattern",
            "torso_pattern",
            "head_pattern",
            "hand_pattern",
            "feet_pattern",
        ):
            self.assertTrue(_matches(cfg.task.profile.body[key], body_names), key)
        for pattern in cfg.task.command.required_motion_body_patterns:
            self.assertTrue(_matches(pattern, body_names), pattern)
        for pattern in cfg.task.command.keypoint_patterns:
            self.assertTrue(_matches(pattern, body_names), pattern)
        self.assertEqual(
            _matches(cfg.task.profile.body.base_pattern, body_names), ["base_link"]
        )
        self.assertEqual(
            _matches(cfg.task.profile.body.torso_pattern, body_names),
            ["waist_yaw_link"],
        )
        self.assertEqual(
            _matches(cfg.task.profile.body.feet_pattern, body_names),
            ["leg_l6_link", "leg_r6_link"],
        )
        self.assertEqual(
            _matches(cfg.task.profile.body.hand_pattern, body_names),
            ["l_handball", "r_handball"],
        )

    def test_action_and_motor_regex_contracts_are_complete_and_nonoverlapping(self):
        cfg = _compose_profile("Roban_tracking")
        joints = ROBAN_JOINT_ORDER

        action_counts = {name: 0 for name in joints}
        action_scale = {}
        for pattern, scale in cfg.task.action.action_scaling.items():
            matched = _matches(pattern, joints)
            self.assertTrue(matched, pattern)
            for name in matched:
                action_counts[name] += 1
                action_scale[name] = scale
        self.assertEqual(set(action_scale), set(joints))
        self.assertTrue(all(count == 1 for count in action_counts.values()))
        self.assertEqual(action_scale["waist_yaw_joint"], 0.5)
        self.assertTrue(
            all(action_scale[name] == 0.5 for name in joints if name.startswith("leg_"))
        )
        self.assertTrue(
            all(action_scale[name] == 1.0 for name in joints if name.startswith("zarm_"))
        )
        self.assertEqual(cfg.task.action.prev_action_obs, "applied")

        motor = cfg.task.randomization.motor_params_implicit
        for field in ("stiffness_range", "damping_range", "armature_range"):
            counts = {name: 0 for name in joints}
            for pattern in motor[field]:
                matched = _matches(pattern, joints)
                self.assertTrue(matched, (field, pattern))
                for name in matched:
                    counts[name] += 1
            self.assertTrue(all(count == 1 for count in counts.values()), field)

    def test_student_contact_and_no_toe_contracts(self):
        cfg = _compose_profile("Roban_tracking")
        self.assertEqual(
            list(cfg.task.command.student_future_steps),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -4, -8],
        )
        self.assertEqual(
            resolve_contact_body_pattern(cfg.task),
            r"^(leg_l6_link|leg_r6_link)$",
        )
        self.assertEqual(
            cfg.task.robot.contact_body_pattern, cfg.task.profile.body.feet_pattern
        )
        dense = cfg.task.reward.loco.feet_air_time_ref_dense
        self.assertFalse(dense.enabled)
        self.assertEqual(dense.body2_names, "__no_feet_toe_body__")
        self.assertNotIn("feet_toe_pattern", cfg.task.profile.body)
        self.assertEqual(cfg.algo.latent_dim, 40)
        self.assertTrue(cfg.algo.student_latent_quantizer.enabled)
        self.assertEqual(
            cfg.algo.student_latent_quantizer.scale, 4.499549865722656
        )

        # A disabled reward must not resolve the deliberately absent toe.
        env = SimpleNamespace(
            scene={"robot": object(), "contact_forces": object()},
            sim=None,
            command_manager=None,
            action_manager=None,
            device=torch.device("cpu"),
            num_envs=2,
        )
        reward = feet_air_time_ref_dense(
            env,
            body_names=dense.body_names,
            body2_names=dense.body2_names,
            enabled=False,
        )
        torch.testing.assert_close(reward.compute(), torch.zeros(2, 1))

    def test_roban_policy_contract_survives_phase_overrides(self):
        for experiment, phase in (("adapt", "adapt"), ("finetune", "finetune")):
            cfg = _compose_profile("Roban_tracking", (f"+exp={experiment}",))
            self.assertEqual(cfg.algo.phase, phase)
            self.assertEqual(cfg.algo.latent_dim, 40)
            self.assertTrue(cfg.algo.student_latent_quantizer.enabled)
            self.assertEqual(
                cfg.algo.student_latent_quantizer.scale, 4.499549865722656
            )

    def test_roban_smoke_phase_configs_are_bounded_and_noise_regexes_match(self):
        expected = (
            ("roban_smoke_train", "train", False),
            ("roban_smoke_adapt", "adapt", True),
            ("roban_smoke_finetune", "finetune", True),
        )
        for experiment, phase, student_train in expected:
            cfg = _compose_profile(
                "Roban_tracking", (f"+exp={experiment}",)
            )
            self.assertEqual(cfg.algo.phase, phase)
            self.assertEqual(cfg.task.num_envs, 64)
            self.assertEqual(cfg.algo.train_every, 8)
            self.assertEqual(cfg.algo.ppo_epochs, 1)
            self.assertEqual(cfg.algo.num_minibatches, 1)
            self.assertEqual(cfg.total_frames, 1024)
            self.assertEqual(cfg.task.student_train, student_train)
            self.assertEqual(cfg.wandb.mode, "disabled")
            self.assertTrue(cfg.headless)
            self.assertFalse(cfg.eval_render)
            self.assertEqual(
                cfg.task.profile.command.dataset.groups[0].teacher_mem_path,
                "_smoke/roban/main_subset",
            )
            for pattern in cfg.algo.init_noise_scale_overrides:
                self.assertTrue(_matches(pattern, ROBAN_JOINT_ORDER), pattern)

    def test_g1_and_l7_keep_profile_driven_contact_and_default_policy_contract(self):
        expected_feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        for profile, robot_cfg in (
            ("G1_tracking", G1_CFG),
            ("L7_tracking", XBOT_L7_CFG),
        ):
            cfg = _compose_profile(profile)
            self.assertEqual(
                resolve_contact_body_pattern(cfg.task), cfg.task.profile.body.feet_pattern
            )
            model = robot_cfg.spec_fn().compile()
            body_names = [model.body(i).name for i in range(model.nbody)]
            self.assertEqual(
                _matches(resolve_contact_body_pattern(cfg.task), body_names),
                expected_feet,
            )
            self.assertTrue(cfg.task.reward.loco.feet_air_time_ref_dense.enabled)
            self.assertEqual(cfg.algo.latent_dim, 256)
            self.assertFalse(cfg.algo.student_latent_quantizer.enabled)
            self.assertEqual(cfg.task.action.prev_action_obs, "mean")

        g1_5_model = G1_CFG_5.spec_fn().compile()
        self.assertEqual(
            _matches(
                _compose_profile("G1_tracking").task.profile.body.feet_pattern,
                [g1_5_model.body(i).name for i in range(g1_5_model.nbody)],
            ),
            expected_feet,
        )

        legacy_cfg = SimpleNamespace(
            robot={}, command={"feet_patterns": ["legacy_feet_pattern"]}
        )
        self.assertEqual(
            resolve_contact_body_pattern(legacy_cfg), "legacy_feet_pattern"
        )
        namespace_cfg = SimpleNamespace(
            robot=SimpleNamespace(contact_body_pattern="namespace_feet")
        )
        self.assertEqual(
            resolve_contact_body_pattern(namespace_cfg), "namespace_feet"
        )

    def test_wrench_body_accepts_profile_regex_and_requires_one_match(self):
        class Asset:
            def find_bodies(self, pattern):
                self.pattern = pattern
                return [3], ["waist_yaw_link"]

        asset = Asset()
        env = SimpleNamespace(
            scene={"robot": asset},
            sim=None,
            command_manager=None,
            action_manager=None,
            device=torch.device("cpu"),
            num_envs=2,
        )
        randomizer = perturb_body_wrench(
            env,
            body_name=r"^waist_yaw_link$",
            total_duration_range_s=(3.0, 6.0),
            active_duration_range_s=(1.0, 4.0),
            force_magnitude_range_n=(0.0, 10.0),
            lever_arm_length_range_m=(0.0, 0.4),
        )
        self.assertEqual(asset.pattern, r"^waist_yaw_link$")
        self.assertEqual(randomizer.body_id, 3)


if __name__ == "__main__":
    unittest.main()
