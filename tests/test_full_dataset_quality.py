import unittest
from pathlib import Path
import numpy as np

from scripts.data_process.build_roban_full_dataset import quality_decision


def motion(frames=200):
    qpos = np.zeros((frames, 28), dtype=np.float32)
    qpos[:, 2] = .8
    qpos[:, 3] = 1
    xpos = np.zeros((frames, 28, 3), dtype=np.float32)
    xpos[:, 0, 2] = .8
    return {'qpos': qpos, 'xpos': xpos, 'fps': 50}


class QualityTest(unittest.TestCase):
    def test_valid_file_kept(self):
        self.assertEqual(quality_decision(motion(), Path('walk.npz')), (True, [], []))

    def test_short_file_rejected(self):
        keep, reasons, _ = quality_decision(motion(149), Path('short.npz'))
        self.assertFalse(keep)
        self.assertIn('short length', reasons)

    def test_one_velocity_spike_rejects_whole_file(self):
        sample = motion()
        sample['qpos'][100:, 7] = 5
        keep, reasons, intervals = quality_decision(sample, Path('spike.npz'))
        self.assertFalse(keep)
        self.assertIn('high joint velocity spike', reasons)
        self.assertTrue(intervals)

    def test_penetration_rejects_whole_file(self):
        sample = motion()
        sample['xpos'][80, 1, 2] = -.2
        keep, reasons, intervals = quality_decision(sample, Path('ground.npz'))
        self.assertFalse(keep)
        self.assertIn('min link z < -0.10m', reasons)


if __name__ == '__main__':
    unittest.main()
