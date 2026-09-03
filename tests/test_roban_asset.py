import re
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from active_adaptation.assets.roban import (
    ROBAN_ARTICULATION,
    ROBAN_CFG,
    ROBAN_COLLISION,
    ROBAN_JOINT_ORDER,
    ROBAN_MJCF_ACTUATOR_ORDER,
    ROBAN_MJCF_JOINT_TREE_ORDER,
    ROBAN_XML,
)
from active_adaptation.assets._auto_symmetry import load_spec_from_xml


EXPECTED_PD = {
    "waist_yaw_joint": (40.1792, 3.5579, 80.0),
    "leg_l1_joint": (90.1792, 3.5579, 150.0),
    "leg_r1_joint": (90.1792, 3.5579, 150.0),
    "leg_l2_joint": (150.0984, 8.3088, 150.0),
    "leg_r2_joint": (150.0984, 8.3088, 150.0),
    "leg_l3_joint": (40.1792, 3.5579, 70.0),
    "leg_r3_joint": (40.1792, 3.5579, 70.0),
    "leg_l4_joint": (150.0984, 8.3088, 150.0),
    "leg_r4_joint": (150.0984, 8.3088, 150.0),
    "leg_l5_joint": (34.2506, 2.9072, 74.0),
    "leg_r5_joint": (34.2506, 2.9072, 74.0),
    "leg_l6_joint": (34.2506, 2.9072, 74.0),
    "leg_r6_joint": (34.2506, 2.9072, 74.0),
    "zarm_l1_joint": (14.2506, 1.9072, 14.1),
    "zarm_r1_joint": (14.2506, 1.9072, 14.1),
    "zarm_l2_joint": (14.2506, 1.9072, 37.0),
    "zarm_l3_joint": (14.2506, 1.9072, 37.0),
    "zarm_l4_joint": (14.2506, 1.9072, 37.0),
    "zarm_r2_joint": (14.2506, 1.9072, 37.0),
    "zarm_r3_joint": (14.2506, 1.9072, 37.0),
    "zarm_r4_joint": (14.2506, 1.9072, 37.0),
}


def _raw_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(ROBAN_XML))


def _active_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        model.joint(i).name
        for i in range(model.njnt)
        if model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE
    )


class RobanAssetTest(unittest.TestCase):
    def test_xml_meshes_and_both_compile_paths(self):
        self.assertTrue(ROBAN_XML.is_file())
        xml_root = ET.parse(ROBAN_XML).getroot()
        mesh_files = [mesh.attrib["file"] for mesh in xml_root.findall("./asset/mesh")]
        self.assertEqual(len(mesh_files), 28)
        self.assertEqual(len(mesh_files), len(set(mesh_files)))
        mesh_dir = ROBAN_XML.parent.parent / "meshes"
        for filename in mesh_files:
            self.assertTrue((mesh_dir / filename).is_file(), filename)
        self.assertEqual(
            {path.name for path in mesh_dir.glob("*.STL")}, set(mesh_files)
        )

        package_root = ET.parse(ROBAN_XML.parent.parent / "package.xml").getroot()
        self.assertEqual(package_root.findtext("version"), "1.0.0")
        self.assertEqual(package_root.findtext("license"), "BSD")

        raw = _raw_model()
        source_spec_model = load_spec_from_xml(ROBAN_XML).compile()
        spec_model = ROBAN_CFG.spec_fn().compile()
        self.assertEqual(
            (raw.nq, raw.nv, raw.nu, raw.njnt, raw.nbody, raw.ngeom),
            (28, 27, 21, 22, 38, 65),
        )
        self.assertEqual(
            (
                source_spec_model.nq,
                source_spec_model.nv,
                source_spec_model.nu,
                source_spec_model.njnt,
                source_spec_model.nbody,
                source_spec_model.ngeom,
            ),
            (28, 27, 21, 22, 38, 65),
        )
        self.assertEqual(spec_model.nq, raw.nq)
        self.assertEqual(spec_model.nv, raw.nv)

    def test_joint_body_mass_and_initial_state_contract(self):
        raw = _raw_model()
        active_joints = _active_joint_names(raw)
        body_names = {raw.body(i).name for i in range(raw.nbody)}

        self.assertEqual(len(ROBAN_JOINT_ORDER), 21)
        self.assertEqual(len(set(ROBAN_JOINT_ORDER)), 21)
        self.assertEqual(set(active_joints), set(ROBAN_JOINT_ORDER))
        self.assertEqual(active_joints, ROBAN_MJCF_JOINT_TREE_ORDER)
        self.assertEqual(ROBAN_MJCF_JOINT_TREE_ORDER, ROBAN_JOINT_ORDER)
        self.assertNotIn("zhead_1_joint", ROBAN_JOINT_ORDER)
        self.assertNotIn("zhead_2_joint", ROBAN_JOINT_ORDER)
        self.assertTrue({"l_handball", "r_handball"}.issubset(body_names))
        self.assertFalse({"l_handball", "r_handball"}.intersection(ROBAN_JOINT_ORDER))
        self.assertEqual(raw.body(1).name, "base_link")
        np.testing.assert_allclose(raw.body_pos[1], [0.0, 0.0, 0.77])
        self.assertAlmostEqual(
            float(raw.body_mass.sum()), 40.69991336, delta=1.0e-5
        )

        entity_model = ROBAN_CFG.build().spec.compile()
        init_key = entity_model.key("init_state")
        self.assertTrue(np.isfinite(init_key.qpos).all())
        np.testing.assert_allclose(init_key.qpos[:3], [0.0, 0.0, 0.77])
        np.testing.assert_allclose(init_key.qpos[7:], [0.0] * 21)
        for name in ROBAN_JOINT_ORDER:
            raw_joint = raw.joint(name)
            dof_id = raw_joint.dofadr
            self.assertAlmostEqual(raw.dof_armature[dof_id], 0.003)
            self.assertAlmostEqual(raw.dof_damping[dof_id], 0.2)
            self.assertAlmostEqual(raw.dof_frictionloss[dof_id], 0.0)
            entity_joint = entity_model.joint(name)
            entity_dof_id = entity_joint.dofadr
            self.assertAlmostEqual(entity_model.dof_armature[entity_dof_id], 0.003)
            self.assertAlmostEqual(entity_model.dof_damping[entity_dof_id], 0.2)
            self.assertAlmostEqual(entity_model.dof_frictionloss[entity_dof_id], 0.0)
            lo, hi = entity_joint.range
            self.assertLessEqual(lo, 0.0, name)
            self.assertGreaterEqual(hi, 0.0, name)

    def test_source_and_training_actuator_orders_are_explicit(self):
        raw = _raw_model()
        raw_targets = tuple(
            raw.joint(int(raw.actuator_trnid[i, 0])).name for i in range(raw.nu)
        )
        self.assertEqual(raw_targets, ROBAN_MJCF_ACTUATOR_ORDER)
        self.assertNotEqual(raw_targets, ROBAN_JOINT_ORDER)

        training_model = ROBAN_CFG.build().spec.compile()
        self.assertEqual(training_model.nu, 21)
        self.assertEqual(
            tuple(training_model.actuator(i).name for i in range(training_model.nu)),
            ROBAN_JOINT_ORDER,
        )
        for index, name in enumerate(ROBAN_JOINT_ORDER):
            self.assertEqual(
                training_model.joint(int(training_model.actuator_trnid[index, 0])).name,
                name,
            )
            _, _, effort = EXPECTED_PD[name]
            np.testing.assert_allclose(
                training_model.actuator_forcerange[index], [-effort, effort]
            )

    def test_training_spec_exposes_generic_imu_sensor_names(self):
        raw = _raw_model()
        self.assertNotEqual(raw.sensor("BodyGyro").id, -1)
        self.assertNotEqual(raw.sensor("BodyAcc").id, -1)

        training_model = ROBAN_CFG.spec_fn().compile()
        self.assertNotEqual(training_model.sensor("imu_ang_vel").id, -1)
        self.assertNotEqual(training_model.sensor("imu_lin_acc").id, -1)
        self.assertEqual(
            training_model.sensor("imu_ang_vel").type,
            mujoco.mjtSensor.mjSENS_GYRO,
        )
        self.assertEqual(
            training_model.sensor("imu_lin_acc").type,
            mujoco.mjtSensor.mjSENS_ACCELEROMETER,
        )

    def test_actuator_groups_cover_policy_joints_once_with_verified_pd(self):
        matches: dict[str, list] = {name: [] for name in ROBAN_JOINT_ORDER}
        for actuator in ROBAN_ARTICULATION.actuators:
            for pattern in actuator.target_names_expr:
                matched = [
                    name for name in ROBAN_JOINT_ORDER if re.fullmatch(pattern, name)
                ]
                self.assertTrue(matched, pattern)
                for name in matched:
                    matches[name].append(actuator)

        self.assertTrue(all(len(groups) == 1 for groups in matches.values()))
        for name, groups in matches.items():
            actuator = groups[0]
            self.assertAlmostEqual(actuator.armature, 0.003)
            np.testing.assert_allclose(
                (actuator.stiffness, actuator.damping, actuator.effort_limit),
                EXPECTED_PD[name],
            )
        self.assertAlmostEqual(ROBAN_ARTICULATION.soft_joint_pos_limit_factor, 0.9)

    def test_collision_config_matches_named_foot_collisions(self):
        model = ROBAN_CFG.build().spec.compile()
        collision_names = [
            model.geom(i).name for i in range(model.ngeom) if model.geom_contype[i] != 0
        ]
        self.assertEqual(len(collision_names), 37)
        self.assertEqual(len(collision_names), len(set(collision_names)))
        self.assertTrue(
            all(
                any(
                    re.fullmatch(pattern, name)
                    for pattern in ROBAN_COLLISION.geom_names_expr
                )
                for name in collision_names
            )
        )

        foot_pattern = next(iter(ROBAN_COLLISION.friction))
        foot_names = [name for name in collision_names if re.fullmatch(foot_pattern, name)]
        self.assertEqual(len(foot_names), 6)
        for name in foot_names:
            self.assertAlmostEqual(model.geom(name).friction[0], 0.6)


if __name__ == "__main__":
    unittest.main()
