import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RobanPipelineTest(unittest.TestCase):
    def run_pipeline(self, directory, **overrides):
        directory = Path(directory)
        runner = directory / 'torchrun'
        runner.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
args['test_ld_library_path'] = os.environ.get('LD_LIBRARY_PATH')
with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')
p = Path(args['hydra.run.dir'])
if args['+exp'] == os.environ.get('FAIL_STAGE'): sys.exit(7)
if args['+exp'] != os.environ.get('MISSING_STAGE'): (p/'checkpoint_final.pt').write_text('mock')
''')
        runner.chmod(0o755)
        env = dict(os.environ, TORCHRUN=str(runner), OUTPUT_ROOT=str(directory/'output'),
                   CALLS=str(directory/'calls'), CUDA_VISIBLE_DEVICES='', NPROC='',
                   PER_RANK_ENVS='', DRY_RUN='0', START_STAGE='train', CHECKPOINT_PATH='',
                   TRAIN_ROLLOUTS='', ADAPT_ROLLOUTS='', FINETUNE_ROLLOUTS='')
        env.update(overrides)
        result = subprocess.run(['bash', str(ROOT/'train.sh')], cwd='/tmp', env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in (directory/'calls').read_text().splitlines()] if (directory/'calls').exists() else []
        return result, calls

    def test_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            result, calls = self.run_pipeline(d, LD_LIBRARY_PATH="/tmp/heft-test-foreign-cuda")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([c['+exp'] for c in calls], ['roban_train', 'roban_adapt', 'roban_finetune'])
            self.assertEqual([c['total_frames'] for c in calls], ['52428800000', '2621440000', '5242880000'])
            for i, c in enumerate(calls):
                self.assertEqual(c['wandb.mode'], 'disabled')
                self.assertIsNone(c['test_ld_library_path'])
                if i: self.assertEqual(c['checkpoint_path'], calls[i-1]['hydra.run.dir']+'/checkpoint_final.pt')

    def test_failures_stop_pipeline(self):
        for failure in ('FAIL_STAGE', 'MISSING_STAGE'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                result, calls = self.run_pipeline(d, **{failure: 'roban_adapt'})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(calls), 2)

    def test_start_finetune(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d)/'existing.pt'
            checkpoint.write_text('mock')
            result, calls = self.run_pipeline(d, START_STAGE='finetune', CHECKPOINT_PATH=str(checkpoint))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['checkpoint_path'], str(checkpoint))

    def test_dry_run_and_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            result, calls = self.run_pipeline(d, DRY_RUN='1')
            self.assertEqual(result.returncode, 0)
            self.assertFalse(calls)
            self.assertFalse((Path(d)/'output').exists())
            (Path(d)/'output').mkdir()
            result, calls = self.run_pipeline(d)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
