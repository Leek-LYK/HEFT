import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.compare_xwbc_refinement import compare


class ComparisonTest(unittest.TestCase):
    def test_constant_error_offset_survives_unequal_step_weights_and_family_resampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths=[Path(temporary)/name for name in ('baseline','candidate')]
            for side,path in enumerate(paths):
                path.mkdir()
                error=1.+side*.125
                rows=[dict(source='human',robot='g1',motion_id=str(i),family_id=str(i//2),
                           success=True,completion=1.,steps=(i+1)*10,reason='complete',
                           errors=dict(mpkpe_root_relative_m=error,mpkpe_world_m=error*2)) for i in range(4)]
                (path/'clips.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                summary=dict(dataset_digest='same',context={},split='val',motion_ids_digest='same',
                             expected_clips=4,full_scope=True,checkpoint_sha256=str(side),
                             matrix={'human->g1':dict(success_rate=1.,completion=1.,recovery_clips=0,
                                                     mpkpe_root_relative_m=error,mpkpe_world_m=error*2)})
                (path/'summary.json').write_text(json.dumps(summary))
            result=compare(*paths)
            self.assertEqual(result['pooled_success_delta_ci95_pp'],[0.,0.])
            self.assertEqual(result['mean_delta_family_ci95']['completion_pp'],[0.,0.])
            np.testing.assert_allclose(result['mean_delta_family_ci95']['mpkpe_root_relative_m'],[.125,.125])
            np.testing.assert_allclose(result['mean_delta_family_ci95']['mpkpe_world_m'],[.25,.25])


if __name__=='__main__':
    unittest.main()
