import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.refinetrack_v01.build_data import family,corrupt
from scripts.refinetrack_v01.preflight import check_pair

class DataContractTests(unittest.TestCase):
    def test_family_groups_mirrors_and_performers(self):
        self.assertEqual(family('main/210531/jump_001__A001_M.npz'),family('main/210531/jump_001__A002.npz'))
        self.assertNotEqual(family('main/210531/jump_002__A001.npz'),family('main/210531/jump_001__A001.npz'))
    def test_same_length_wrong_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for side,identity in [('clean','a'),('raw','b')]:
                p=root/side;p.mkdir();(p/'id_label.json').write_text(json.dumps([{'source_path':identity}]))
                (p/'meta_motion.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'identity'):check_pair(root/'clean',root/'raw')
    def test_corruption_replay_and_joint_isolation(self):
        for kind in ['jitter','burst','offset','correlated_noise','jump','drift','hold','seam']:
            with self.subTest(kind=kind):
                q=np.repeat(np.linspace(-.5,.5,100)[:,None],21,axis=1).astype(np.float16)
                args=(q,np.full(21,.05),6,[1,2,3],kind,2);limits=np.tile([-.8,.8],(21,1))
                a,event=corrupt(*args,np.random.default_rng(7),limits);b,event2=corrupt(*args,np.random.default_rng(7),limits)
                self.assertTrue(np.array_equal(a,b));self.assertEqual(event,event2)
                self.assertTrue(np.array_equal(a[:,4:],q[:,4:]) and np.array_equal(a[:,0],q[:,0]))
                self.assertTrue(np.isfinite(a).all() and np.any(a!=q))
class MetricTests(unittest.TestCase):
    def test_failure_wins_over_timeout(self):
        from scripts.refinetrack_v01.metrics import completion_result
        self.assertEqual(completion_result(10,10,True,['motion_timeout','body_z_termination']),(False,1.))
    def test_protocol_censoring_is_not_success(self):
        from scripts.refinetrack_v01.metrics import completion_result
        self.assertEqual(completion_result(10,100,False,['episode_timeout']),(False,.1))
    def test_last_frame_required(self):
        from scripts.refinetrack_v01.metrics import completion_result
        self.assertFalse(completion_result(98,99,False,['motion_timeout'])[0])
        self.assertTrue(completion_result(99,99,False,['motion_timeout'])[0])
if __name__=='__main__':unittest.main()
