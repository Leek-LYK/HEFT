import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scripts.data_process.convert_roban_planner_csv import INPUT_FIELDS, JOINT_NAMES, convert_motion


class PlannerCSVTest(unittest.TestCase):
    def source(self, root, bad_time=False):
        path=Path(root)/'motion.csv'
        fields=list(reversed(INPUT_FIELDS))
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i in range(3):
                row={k:0. for k in INPUT_FIELDS};row['timestamp']=i*.04
                row['root_translateZ']=.75;row['root_translateX']=i*.04
                row['root_quat_w']=-1. if i==1 else 1.
                for j,name in enumerate(JOINT_NAMES):row[name+'_dof']=j*.01+i*.04
                if bad_time and i==2:row['timestamp']=.01
                writer.writerow(row)
        return path

    def test_named_mapping_resampling_and_velocity(self):
        with tempfile.TemporaryDirectory() as d:
            out,report=convert_motion(self.source(d),'chest')
            self.assertEqual(out.shape,(5,49))
            np.testing.assert_allclose(out[:,7:28],np.arange(21)[None,:]*.01+np.arange(5)[:,None]*.02,atol=1e-14)
            np.testing.assert_allclose(out[:,28:],1.,atol=1e-14)
            np.testing.assert_allclose(np.abs(out[:,3]),1.)
            self.assertEqual(report['timestamp_span_s'],.08)

    def test_explicit_pelvis_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            out,_=convert_motion(self.source(d),'pelvis')
            np.testing.assert_allclose(out[:,7],-np.arange(5)*.02)
            rot=Rotation.from_quat(np.roll(out[:,3:7],-1,axis=1))
            np.testing.assert_allclose(rot.as_rotvec()[:,2],np.arange(5)*.02,atol=1e-14)
            np.testing.assert_allclose(out[:,28],-1.)

    def test_invalid_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):convert_motion(self.source(d,True),'chest')


if __name__=='__main__':unittest.main()
