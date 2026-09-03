import unittest

from active_adaptation.assets._auto_symmetry import (
    DEFAULT_NAME_SWAP_RULES,
    base_joint_sign,
    build_auto_joint_symmetry_map,
    build_auto_spatial_symmetry_map,
    parse_xml_symmetry_info,
    swap_left_right,
)
from active_adaptation.assets.g1 import G1_XML, JOINT_SYMMETRY_MAP as G1_JOINT_MAP
from active_adaptation.assets.roban import (
    JOINT_SYMMETRY_MAP,
    ROBAN_JOINT_ORDER,
    ROBAN_NAME_SWAP_RULES,
    ROBAN_XML,
    SPATIAL_SYMMETRY_MAP,
)
from active_adaptation.assets.xbot_l7 import (
    JOINT_SYMMETRY_MAP as L7_JOINT_MAP,
    XBOT_L7_XML,
)


def _legacy_default_joint_map(joint_axes):
    """Reproduce the pre-Roban default mapping to guard G1/L7 behavior."""
    joint_names = set(joint_axes)
    mapping = {}
    for name, axis in joint_axes.items():
        mirrored = swap_left_right(name)
        if mirrored not in joint_names:
            mirrored = name
        sign = base_joint_sign(name, axis)
        if mirrored != name:
            mirrored_axis = joint_axes[mirrored]
            if sum(a * b for a, b in zip(axis, mirrored_axis)) < 0.0:
                sign = -sign
        mapping[name] = (sign, mirrored)
    return mapping


def _legacy_default_spatial_map(body_names):
    body_name_set = set(body_names)
    mapping = {}
    for name in body_names:
        mirrored = swap_left_right(name)
        mapping[name] = mirrored if mirrored in body_name_set else name
    return mapping


class RobanSymmetryTest(unittest.TestCase):
    def test_roban_name_rules_and_maps_are_involutions(self):
        _, _, body_names = parse_xml_symmetry_info(ROBAN_XML)
        for name in (*ROBAN_JOINT_ORDER, *body_names):
            self.assertEqual(
                swap_left_right(
                    swap_left_right(name, ROBAN_NAME_SWAP_RULES),
                    ROBAN_NAME_SWAP_RULES,
                ),
                name,
            )

        for name, (_, mirrored) in JOINT_SYMMETRY_MAP.items():
            self.assertEqual(JOINT_SYMMETRY_MAP[mirrored][1], name)
        for name, mirrored in SPATIAL_SYMMETRY_MAP.items():
            self.assertEqual(SPATIAL_SYMMETRY_MAP[mirrored], name)

    def test_roban_leg_arm_hand_and_center_mappings(self):
        for index in range(1, 7):
            self.assertEqual(
                JOINT_SYMMETRY_MAP[f"leg_l{index}_joint"][1],
                f"leg_r{index}_joint",
            )
            self.assertEqual(
                JOINT_SYMMETRY_MAP[f"leg_r{index}_joint"][1],
                f"leg_l{index}_joint",
            )
        for index in range(1, 5):
            self.assertEqual(
                JOINT_SYMMETRY_MAP[f"zarm_l{index}_joint"][1],
                f"zarm_r{index}_joint",
            )
            self.assertEqual(
                JOINT_SYMMETRY_MAP[f"zarm_r{index}_joint"][1],
                f"zarm_l{index}_joint",
            )
        self.assertEqual(
            JOINT_SYMMETRY_MAP["waist_yaw_joint"][1], "waist_yaw_joint"
        )
        self.assertEqual(SPATIAL_SYMMETRY_MAP["l_handball"], "r_handball")
        self.assertEqual(SPATIAL_SYMMETRY_MAP["r_handball"], "l_handball")
        for name in ("base_link", "waist_yaw_link", "zhead_1_link", "zhead_2_link"):
            self.assertEqual(SPATIAL_SYMMETRY_MAP[name], name)

    def test_default_g1_l7_symmetry_generation_is_unchanged(self):
        for xml_path, expected_joint_map in (
            (G1_XML, G1_JOINT_MAP),
            (XBOT_L7_XML, L7_JOINT_MAP),
        ):
            axes, _, body_names = parse_xml_symmetry_info(xml_path)
            legacy_joint_map = _legacy_default_joint_map(axes)
            self.assertEqual(expected_joint_map, legacy_joint_map)
            self.assertEqual(
                build_auto_joint_symmetry_map(axes, DEFAULT_NAME_SWAP_RULES),
                legacy_joint_map,
            )
            spatial_map = build_auto_spatial_symmetry_map(
                body_names, DEFAULT_NAME_SWAP_RULES
            )
            self.assertEqual(spatial_map, _legacy_default_spatial_map(body_names))
            self.assertTrue(
                all(spatial_map[spatial_map[name]] == name for name in body_names)
            )


if __name__ == "__main__":
    unittest.main()
