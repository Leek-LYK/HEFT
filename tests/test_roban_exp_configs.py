import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import active_adaptation.learning  # noqa: F401 - registers structured algo configs
from active_adaptation.assets.roban import ROBAN_JOINT_ORDER


CFG_DIR = Path(__file__).parents[1] / "cfg"

ROBAN_NOISE_GROUPS = (
    ("waist", r"^waist_yaw_joint$", 1.5),
    ("hip_chain", r"^leg_[lr][123]_joint$", 1.5),
    ("knee", r"^leg_[lr]4_joint$", 1.5),
    ("ankle", r"^leg_[lr][56]_joint$", 1.0),
    ("shoulder_chain", r"^zarm_[lr][123]_joint$", 1.2),
    ("elbow", r"^zarm_[lr]4_joint$", 1.2),
)
ROBAN_NOISE_MAP = {pattern: scale for _, pattern, scale in ROBAN_NOISE_GROUPS}


def _compose(*overrides: str):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
            cfg = compose(config_name="train", overrides=list(overrides))
        OmegaConf.resolve(cfg)
        return cfg
    finally:
        GlobalHydra.instance().clear()


def _matches(pattern: str, names=ROBAN_JOINT_ORDER) -> list[str]:
    return [name for name in names if re.fullmatch(pattern, name)]


def _assert_exact_partition(mapping: Mapping, names=ROBAN_JOINT_ORDER) -> None:
    counts = {name: 0 for name in names}
    for pattern in mapping:
        matched = _matches(pattern, names)
        assert matched, f"joint regex matches nothing: {pattern}"
        for name in matched:
            counts[name] += 1
    assert all(count == 1 for count in counts.values()), counts


def _value_for_joint(mapping: Mapping, joint_name: str):
    values = [value for pattern, value in mapping.items() if re.fullmatch(pattern, joint_name)]
    assert len(values) == 1, (joint_name, values)
    return values[0]


@pytest.fixture(scope="module")
def formal_configs():
    return {
        phase: _compose(f"+exp=roban_{phase}")
        for phase in ("train", "adapt", "finetune")
    }


def test_roban_formal_configs_compose_with_profile_and_official_phases(formal_configs):
    expected = {
        "train": ("train", "train", False),
        "adapt": ("adapt", "eval", True),
        "finetune": ("finetune", "eval", True),
    }
    for name, cfg in formal_configs.items():
        phase, vecnorm, student_train = expected[name]
        assert cfg.task.profile.name == "roban"
        assert cfg.task.robot.name == "roban"
        assert cfg.algo.phase == phase
        assert cfg.algo.vecnorm == vecnorm
        assert cfg.task.student_train is student_train
        assert cfg.task.recover_nonfinite_state is True
        assert "nonfinite_state" in cfg.task.termination


def test_roban_formal_configs_reuse_official_experiment_settings(formal_configs):
    ignored_roban_policy_fields = {
        "init_noise_scale_overrides",
        "latent_dim",
        "student_latent_quantizer",
    }
    for name, cfg in formal_configs.items():
        official = _compose(f"+exp={name}")
        actual_algo = OmegaConf.to_container(cfg.algo, resolve=True)
        official_algo = OmegaConf.to_container(official.algo, resolve=True)
        for field in ignored_roban_policy_fields:
            actual_algo.pop(field)
            official_algo.pop(field)
        assert actual_algo == official_algo
        assert cfg.total_frames == official.total_frames
        assert cfg.task.student_train == official.task.student_train
        if name == "train":
            assert list(cfg.task.reward.loco._scale_factor_) == list(
                official.task.reward.loco._scale_factor_
            )
        if name == "finetune":
            assert cfg.task.reward.loco.action_rate_l2 == official.task.reward.loco.action_rate_l2


def test_roban_formal_configs_do_not_embed_smoke_runtime_settings(formal_configs):
    for cfg in formal_configs.values():
        resolved_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
        assert cfg.total_frames != 1024
        assert cfg.algo.train_every != 8
        assert cfg.algo.ppo_epochs != 1
        assert cfg.algo.num_minibatches != 1
        assert cfg.task.num_envs != 64
        assert "_smoke/roban/main_subset" not in resolved_json
        assert "smoke" not in cfg
        assert cfg.wandb.mode == "online"


def test_g1_l7_default_recovery_and_event_diagnostics_remain_disabled():
    for profile in ("G1_tracking", "L7_tracking"):
        cfg = _compose(f"task/profile={profile}")
        assert bool(cfg.task.get("recover_nonfinite_state", False)) is False
        assert "nonfinite_state" not in cfg.task.termination
        assert "nonfinite_diagnostics" not in cfg.task
        assert "pilot" not in cfg


def test_roban_quantized_latent_contract_survives_all_formal_phases(formal_configs):
    for cfg in formal_configs.values():
        assert cfg.algo.latent_dim == 40
        assert cfg.algo.student_latent_quantizer.enabled is True
        assert cfg.algo.student_latent_quantizer.scale == pytest.approx(4.499549865722656)


def test_train_and_finetune_action_noise_is_a_complete_roban_partition(formal_configs):
    for phase in ("train", "finetune"):
        mapping = formal_configs[phase].algo.init_noise_scale_overrides
        assert dict(mapping) == ROBAN_NOISE_MAP
        _assert_exact_partition(mapping)
        assert all("wrist" not in pattern for pattern in mapping)

    assert dict(formal_configs["adapt"].algo.init_noise_scale_overrides) == {}


def test_all_roban_joint_noise_and_motor_regex_maps_are_nonempty_partitions(formal_configs):
    cfg = formal_configs["train"]
    joint_maps = {
        "policy.init_noise_scale_overrides": cfg.task.profile.policy.init_noise_scale_overrides,
        "student_motion_randomization.joint_pos_bias_std": (
            cfg.task.command.student_motion_randomization.joint_pos_bias_std
        ),
        "shared_joint_pos.std": cfg.task.command.shared_joint_pos.std,
        "shared_joint_pos.bias": cfg.task.command.shared_joint_pos.bias,
        "shared_joint_vel.std": cfg.task.command.shared_joint_vel.std,
        "motor.stiffness_range": cfg.task.randomization.motor_params_implicit.stiffness_range,
        "motor.damping_range": cfg.task.randomization.motor_params_implicit.damping_range,
        "motor.armature_range": cfg.task.randomization.motor_params_implicit.armature_range,
        "random_joint_offset": cfg.task.randomization.random_joint_offset,
    }
    for name, mapping in joint_maps.items():
        try:
            _assert_exact_partition(mapping)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from error
        assert all("wrist" not in pattern for pattern in mapping), name


def test_roban_noise_values_are_preserved_from_official_train_and_l7():
    roban = _compose("+exp=roban_train")
    official_train = _compose("+exp=train")
    l7 = _compose("task/profile=L7_tracking")

    semantic_pairs = {
        "waist_yaw_joint": "waist_yaw_joint",
        "leg_l1_joint": "left_hip_pitch_joint",
        "leg_l4_joint": "left_knee_joint",
        "leg_l5_joint": "left_ankle_pitch_joint",
        "zarm_l1_joint": "left_shoulder_pitch_joint",
        "zarm_l4_joint": "left_elbow_pitch_joint",
    }
    for roban_joint, official_joint in semantic_pairs.items():
        assert _value_for_joint(
            roban.algo.init_noise_scale_overrides, roban_joint
        ) == _value_for_joint(official_train.algo.init_noise_scale_overrides, official_joint)

    assert (
        roban.task.profile.observation.policy.noise
        == l7.task.profile.observation.policy.noise
    )
    for field in ("stiffness_range", "damping_range", "armature_range"):
        roban_ranges = roban.task.randomization.motor_params_implicit[field]
        l7_ranges = l7.task.randomization.motor_params_implicit[field]
        for roban_joint, l7_joint in semantic_pairs.items():
            assert list(_value_for_joint(roban_ranges, roban_joint)) == list(
                _value_for_joint(l7_ranges, l7_joint)
            )

    assert "frictionloss_range" not in roban.task.randomization.motor_params_implicit


def test_no_g1_only_joint_regex_or_fake_wrist_remains_in_resolved_roban_configs(
    formal_configs,
):
    g1_only_joint_tokens = ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")
    for cfg in formal_configs.values():
        resolved = OmegaConf.to_yaml(cfg, resolve=True)
        for token in g1_only_joint_tokens:
            assert token not in resolved
        assert all("wrist" not in joint for joint in ROBAN_JOINT_ORDER)


def test_existing_g1_g1_5_and_xbot_l7_configurations_still_compose():
    configs = (
        (_compose("task/profile=G1_tracking"), "g1"),
        (_compose("task/profile=G1_tracking", "task.profile.name=g1_5"), "g1_5"),
        (_compose("task/profile=L7_tracking"), "xbot_l7"),
    )
    for cfg, expected_robot_name in configs:
        assert cfg.task.robot.name == expected_robot_name
        assert cfg.algo.phase == "train"
