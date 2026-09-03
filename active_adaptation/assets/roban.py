from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from ._auto_symmetry import (
    DEFAULT_NAME_SWAP_RULES,
    generate_auto_symmetry,
    load_spec_from_xml,
)

ASSET_PATH = os.path.dirname(__file__)
ROBAN_XML = (
    Path(ASSET_PATH)
    / "roban"
    / "mjcf"
    / "biped_s17_verified_handball_fixedhead.xml"
)


def _get_spec() -> mujoco.MjSpec:
    """Load the Roban MJCF for position control without changing its physics."""
    spec = load_spec_from_xml(ROBAN_XML)

    # The Roban source uses deployment-facing IMU names. Expose the generic
    # names consumed by HEFT observations while retaining the same sensor
    # types, sites, and measurements.
    sensor_name_map = {
        "BodyGyro": "imu_ang_vel",
        "BodyAcc": "imu_lin_acc",
    }
    for sensor in spec.sensors:
        if sensor.name in sensor_name_map:
            sensor.name = sensor_name_map[sensor.name]

    # The source MJCF motors document its actuator order and effort limits. The
    # training entity replaces them with the position actuators configured below.
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    # Source collision geoms are unnamed. Give every geom a deterministic name
    # in memory so CollisionCfg can select them without editing geometry data.
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for geom in spec.geoms:
        body_name = geom.parent.name
        kind = "collision" if geom.contype or geom.conaffinity else "visual"
        key = (body_name, kind)
        index = counts[key]
        counts[key] += 1
        geom.name = f"{body_name}_{kind}_{index}"
    return spec


ROBAN_INIT_STATE = EntityCfg.InitialStateCfg(
    # Verified from the authoritative MJCF floating-root body position.
    pos=(0.0, 0.0, 0.77),
    # The deployed HEFT policy and every actuated MJCF range contain zero.
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


ROBAN_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^waist_yaw_joint$",),
    armature=0.003,
    stiffness=40.1792,
    damping=3.5579,
    effort_limit=80.0,
)
ROBAN_ACTUATOR_LEG_1 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^leg_[lr]1_joint$",),
    armature=0.003,
    stiffness=90.1792,
    damping=3.5579,
    effort_limit=150.0,
)
ROBAN_ACTUATOR_LEG_2 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^leg_[lr]2_joint$",),
    armature=0.003,
    stiffness=150.0984,
    damping=8.3088,
    effort_limit=150.0,
)
ROBAN_ACTUATOR_LEG_3 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^leg_[lr]3_joint$",),
    armature=0.003,
    stiffness=40.1792,
    damping=3.5579,
    effort_limit=70.0,
)
ROBAN_ACTUATOR_LEG_4 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^leg_[lr]4_joint$",),
    armature=0.003,
    stiffness=150.0984,
    damping=8.3088,
    effort_limit=150.0,
)
ROBAN_ACTUATOR_LEG_5 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^leg_[lr]5_joint$",),
    armature=0.003,
    stiffness=34.2506,
    damping=2.9072,
    effort_limit=74.0,
)
ROBAN_ACTUATOR_LEG_6 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^leg_[lr]6_joint$",),
    armature=0.003,
    stiffness=34.2506,
    damping=2.9072,
    effort_limit=74.0,
)
ROBAN_ACTUATOR_ARM_1 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^zarm_[lr]1_joint$",),
    armature=0.003,
    stiffness=14.2506,
    damping=1.9072,
    effort_limit=14.1,
)
ROBAN_ACTUATOR_ARM_2_TO_4 = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^zarm_[lr][2-4]_joint$",),
    armature=0.003,
    stiffness=14.2506,
    damping=1.9072,
    effort_limit=37.0,
)

ROBAN_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        ROBAN_ACTUATOR_WAIST_YAW,
        ROBAN_ACTUATOR_LEG_1,
        ROBAN_ACTUATOR_LEG_2,
        ROBAN_ACTUATOR_LEG_3,
        ROBAN_ACTUATOR_LEG_4,
        ROBAN_ACTUATOR_LEG_5,
        ROBAN_ACTUATOR_LEG_6,
        ROBAN_ACTUATOR_ARM_1,
        ROBAN_ACTUATOR_ARM_2_TO_4,
    ),
    # First-version provisional value: follows the existing framework convention.
    soft_joint_pos_limit_factor=0.9,
)

ROBAN_COLLISION = CollisionCfg(
    geom_names_expr=(r"^.*_collision_[0-9]+$",),
    condim=3,
    priority=1,
    # First-version provisional coefficient; the pattern is asset-verified.
    friction={r"^(leg_l6_link|leg_r6_link)_collision_[0-9]+$": (0.6,)},
)

ROBAN_JOINT_ORDER = (
    "waist_yaw_joint",
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
)

# Kept explicit even though this asset's hinge tree currently happens to match
# policy order. The source torque-actuator order does not.
ROBAN_MJCF_JOINT_TREE_ORDER = (
    "waist_yaw_joint",
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
)
ROBAN_MJCF_ACTUATOR_ORDER = (
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
)

ROBAN_NAME_SWAP_RULES = DEFAULT_NAME_SWAP_RULES + (
    ("leg_l", "leg_r"),
    ("zarm_l", "zarm_r"),
    ("l_handball", "r_handball"),
)
ROBAN_PHYSICAL_BODY_PATTERN = (
    r"^(base_link|waist_yaw_link|leg_[lr][1-6]_link|zarm_[lr][1-4]_link|"
    r"[lr]_handball|zhead_[12]_link|head_radar|head_camera_base)$"
)

JOINT_SYMMETRY_MAP, SPATIAL_SYMMETRY_MAP = generate_auto_symmetry(
    xml_path=ROBAN_XML,
    spec_fn=_get_spec,
    body_pos_tol=1.0e-6,
    body_quat_tol=1.0e-6,
    fk_body_pos_tol=1.0e-6,
    fk_body_quat_tol=1.0e-6,
    name_swap_rules=ROBAN_NAME_SWAP_RULES,
    # Camera optical frames are intentionally asymmetric sensor frames, not
    # policy/motion bodies. Physical links remain exactly symmetric.
    validation_body_patterns=(ROBAN_PHYSICAL_BODY_PATTERN,),
)

ROBAN_CFG = EntityCfg(
    init_state=ROBAN_INIT_STATE,
    collisions=(ROBAN_COLLISION,),
    spec_fn=_get_spec,
    articulation=ROBAN_ARTICULATION,
    sort_actuators=True,
)
ROBAN_CFG.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
ROBAN_CFG.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP
ROBAN_CFG.joint_name_order = ROBAN_JOINT_ORDER
