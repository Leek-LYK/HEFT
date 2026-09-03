import os
from .g1 import G1_CFG, G1_CFG_5
from .roban import ROBAN_CFG
from .xbot_l7 import XBOT_L7_CFG

ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CFG,
    "g1_5": G1_CFG_5,
    "roban": ROBAN_CFG,
    "xbot_l7": XBOT_L7_CFG,
}


def get_robot_cfg(name: str):
    if name not in ROBOTS:
        raise ValueError(f"Unknown robot name: {name}")
    return ROBOTS[name]
