"""CPU characterization of the legacy six-route evaluation protocol.

Real evaluation/aggregation/metrics run on synthetic tensors. Scripted returned
transitions cannot establish simulator independence or recovery physics.
"""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from active_adaptation.envs.xwbc import XWBCPool
from active_adaptation.learning import xwbc as evaluation
from active_adaptation.utils.xwbc_data import CommandBank, ROBOTS, SOURCES, sha256


ERROR_KEYS = ('mpkpe_world_m', 'mpkpe_root_relative_m', 'velocity_mps',
              'root_rotation_rad', 'root_position_m')


class ScriptedPool:
    """Supply events/readings without constructing a simulator."""
    def __init__(self, robot, bank, num_envs, history, evaluation=False):
        assert evaluation
        self.bank, self.num_envs = bank, num_envs
        self.all_ids = torch.arange(num_envs)
        self.fixed_source = None
        self.command = SimpleNamespace(fixed_motion_ids=None,
                                       t=torch.zeros(num_envs, dtype=torch.long),
                                       lengths=torch.zeros(num_envs, dtype=torch.long))
        self.env = SimpleNamespace(_nonfinite_state_mask=torch.zeros(num_envs, dtype=torch.bool),
                                   termination_funcs={'pose': None, 'joint': None})
        self.resolved_config = {'fixture': 'CPU scripted transitions'}
        self.closed = False
        self.resets = []

    def reset(self, ids):
        self.command.t.zero_()
        self.command.lengths.copy_(self.bank.lengths[self.command.fixed_motion_ids])
        self.env._nonfinite_state_mask.zero_()
        self.resets.append((self.fixed_source, self.command.fixed_motion_ids.tolist()))

    def observe(self):
        return dict(proprio=torch.zeros(self.num_envs, 1),
                    motion=self.command.fixed_motion_ids.clone(), frame=self.command.t.clone(),
                    source=torch.full((self.num_envs,), self.fixed_source, dtype=torch.long))

    def step(self, action, mean):
        torch.testing.assert_close(action, mean)
        torch.testing.assert_close(action[:, 0], torch.full((self.num_envs,), float(self.fixed_source)))
        self.command.t += 1
        terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        stats = {name: torch.zeros(self.num_envs) for name in self.env.termination_funcs}
        self.env._nonfinite_state_mask.zero_()
        for i, motion in enumerate(self.command.fixed_motion_ids.tolist()):
            row = self.bank.rows[motion]
            if int(self.command.t[i]) != row.get('stop_at', row['frames'] - 1):
                continue
            event = row.get('event', 'complete')
            if 'ordinary' in event:
                terminated[i] = True
                for name in row.get('reasons', ('pose',)):
                    stats[name][i] = 1
            if 'timeout' in event:
                truncated[i] = True
            if 'recovery' in event:
                terminated[i] = True
                self.env._nonfinite_state_mask[i] = True
        result = dict(done=terminated | truncated, terminated=terminated, truncated=truncated)
        result.update({('stats', 'termination', name): value for name, value in stats.items()})
        return result, self.observe()

    def metrics(self):
        values = []
        for i, motion in enumerate(self.command.fixed_motion_ids.tolist()):
            row = self.bank.rows[motion]
            sequence = row.get('metric_values', [row.get('error', 2.)])
            step = max(0, int(self.command.t[i]) - 1)
            values.append(sequence[min(step, len(sequence) - 1)])
        return {key: torch.tensor(values) for key in ERROR_KEYS}

    def close(self):
        self.closed = True


class XWBCEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='xwbc_semantics_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def row(self, name, **kwargs):
        return dict(motion_id=name, family_id=name, category='main', frames=4, **kwargs)

    def run_evaluation(self, rows, *, num_envs=2, max_motions=None):
        bank = SimpleNamespace(rows=rows, device='cpu',
                               lengths=torch.tensor([r['frames'] for r in rows]))
        bank.gather = lambda motion, frame: {'synthetic': frame.clone()}
        pools, routes = [], []

        def make_pool(*args, **kwargs):
            pool = ScriptedPool(*args, **kwargs)
            pools.append(pool)
            return pool

        def actor(robot, proprio, source, commands):
            routes.append((robot, source.tolist()))
            return SimpleNamespace(mean=source.float()[:, None]), None

        policy = SimpleNamespace(actor=actor, eval=lambda: None)
        state = dict(config={'dataset': 'synthetic-only', 'history': 2}, dataset_digest='synthetic-digest')
        output = self.root / f'eval_{len(list(self.root.glob("eval_*")))}'
        args = SimpleNamespace(seed=42, output=str(output), checkpoint='synthetic-only.pt',
                               dataset=None, split='val', num_envs=num_envs, max_motions=max_motions)
        with contextlib.ExitStack() as stack:
            runtime = stack.enter_context(patch.object(evaluation, 'runtime_setup'))
            stack.enter_context(patch.object(evaluation, 'seed_all'))
            loader = stack.enter_context(patch.object(evaluation, 'load_policy', return_value=(policy, state)))
            constructor = stack.enter_context(patch.object(evaluation, 'CommandBank', return_value=bank))
            stack.enter_context(patch.object(evaluation, 'sha256', return_value='synthetic-digest'))
            stack.enter_context(patch.object(evaluation, 'XWBCPool', side_effect=make_pool))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            evaluation.evaluate(args)
            runtime.assert_called_once_with()
            loader.assert_called_once_with('synthetic-only.pt')
            constructor.assert_called_once_with('synthetic-only', 'val', 'cuda:0')
        self.assertTrue(all(p.closed for p in pools))
        self.assertEqual({(r, s) for r, sources in routes for s in sources},
                         {(r, s) for r in ROBOTS for s in range(len(SOURCES))})
        clips = [json.loads(line) for line in (output / 'clips.jsonl').read_text().splitlines()]
        summary = json.loads((output / 'summary.json').read_text())
        with (output / 'matrix.csv').open() as stream:
            matrix_csv = list(csv.DictReader(stream))
        expected_routes = {f'{s}->{r}' for r in ROBOTS for s in SOURCES}
        self.assertEqual(len(matrix_csv), 6)
        self.assertEqual({row['route'] for row in matrix_csv}, expected_routes)
        self.assertEqual(set(summary['matrix']), expected_routes)
        for clip in clips:
            self.assertEqual(set(clip['errors']), set(ERROR_KEYS))
        for row in matrix_csv:
            # Legacy CSV omits root_position_m, which remains in JSON outputs.
            self.assertNotIn('root_position_m', row)
            for key in ERROR_KEYS[:-1]:
                self.assertAlmostEqual(float(row[key]), summary['matrix'][row['route']][key])
        return clips, summary, matrix_csv, pools

    def test_final_frame_completion_priority_on_all_six_routes(self):
        events = ('complete', 'ordinary', 'timeout', 'recovery',
                  'ordinary_recovery', 'timeout_recovery')
        clips, summary, _, _ = self.run_evaluation([self.row(e, event=e) for e in events])
        self.assertEqual(len(clips), 36)
        for clip in clips:
            with self.subTest(route=(clip['source'], clip['robot']), event=clip['motion_id']):
                recovered = 'recovery' in clip['motion_id']
                self.assertEqual(clip['success'], not recovered)
                self.assertEqual(clip['reason'], 'nonfinite' if recovered else 'complete')
                self.assertEqual(clip['steps'], 3)
                self.assertEqual(clip['completion'], 1.)
        for metrics in summary['matrix'].values():
            self.assertEqual(metrics['success_rate'], .5)
            self.assertEqual(metrics['completion'], 1.)
            self.assertEqual(metrics['recovery_clips'], 3)

    def test_early_termination_timeout_and_recovery_are_failures(self):
        rows = [self.row('ordinary', event='ordinary', stop_at=1, reasons=('pose', 'joint')),
                self.row('timeout', event='timeout', stop_at=1),
                self.row('recovery', event='recovery', stop_at=1)]
        clips, _, _, _ = self.run_evaluation(rows)
        expected = {'ordinary': 'pose,joint', 'timeout': '', 'recovery': 'nonfinite'}
        self.assertEqual(len(clips), 18)
        for clip in clips:
            self.assertFalse(clip['success'])
            self.assertEqual(clip['steps'], 1)
            self.assertAlmostEqual(clip['completion'], 1 / 3)
            self.assertEqual(clip['reason'], expected[clip['motion_id']])
            self.assertEqual(clip['errors']['mpkpe_world_m'], 2.)

    def test_terminal_recovery_reading_is_included_in_legacy_error(self):
        clips, summary, _, _ = self.run_evaluation([
            self.row('recovered', event='recovery', stop_at=2, metric_values=[2., 20.])])
        for clip in clips:
            self.assertFalse(clip['success'])
            self.assertEqual(clip['reason'], 'nonfinite')
            self.assertEqual(clip['steps'], 2)
            self.assertAlmostEqual(clip['completion'], 2 / 3)
            self.assertEqual(clip['errors'], {k: 11. for k in ERROR_KEYS})
        for metrics in summary['matrix'].values():
            self.assertEqual(metrics['mpkpe_world_m'], 11.)

    def test_finished_and_padded_rows_do_not_enter_accumulators(self):
        rows = [self.row('early', event='ordinary', stop_at=1, metric_values=[2., 1000.]),
                self.row('long', error=4.), self.row('tail', error=7.)]
        clips, summary, _, pools = self.run_evaluation(rows, num_envs=2)
        self.assertEqual(len(clips), 18)
        for robot in ROBOTS:
            for source in SOURCES:
                route = [c for c in clips if (c['robot'], c['source']) == (robot, source)]
                self.assertEqual({c['motion_id'] for c in route}, {'early', 'long', 'tail'})
                self.assertEqual(len(route), 3)
                early = next(c for c in route if c['motion_id'] == 'early')
                self.assertEqual(early['steps'], 1)
                self.assertEqual(early['errors']['mpkpe_world_m'], 2.)
                self.assertAlmostEqual(summary['matrix'][f'{source}->{robot}']['mpkpe_world_m'],
                                       (2. + 4. * 3 + 7. * 3) / 7)
        for pool in pools:
            self.assertEqual(pool.resets, [(s, ids) for s in range(3) for ids in ([0, 1], [2, 2])])
        # Output masking only: every mock row still steps. Cross-world effects
        # of inactive rows in the actual simulator require GPU evidence.

    def test_aggregate_weights_errors_by_effective_steps_not_clips(self):
        rows = [dict(family_id='long', success=True, completion=1., reason='complete',
                     steps=9, errors={k: 4. for k in ERROR_KEYS}),
                dict(family_id='short', success=False, completion=.1, reason='pose',
                     steps=1, errors={k: 1. for k in ERROR_KEYS})]
        result = evaluation.aggregate(rows)
        self.assertEqual(result['clips'], 2)
        self.assertEqual(result['families'], 2)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(result['completion'], .55)
        for key in ERROR_KEYS:
            self.assertAlmostEqual(result[key], 3.7)
            self.assertNotEqual(result[key], 2.5)
        # Low error on a short failed prefix is not full-clip quality.
        failed = evaluation.aggregate(rows[1:])
        self.assertEqual(failed['mpkpe_world_m'], 1.)
        self.assertEqual(failed['success_rate'], 0.)
        self.assertEqual(failed['completion'], .1)

    def test_family_bootstrap_keeps_clip_weighting_and_is_repeatable(self):
        rows = [dict(family_id=f, success=s, completion=float(s), reason='complete' if s else 'pose',
                     steps=1, errors={'mpkpe_world_m': 1.})
                for f, s in [('a', True), ('a', True), ('b', False)]]
        result = evaluation.aggregate(rows)
        self.assertAlmostEqual(result['success_rate'], 2 / 3)
        self.assertEqual(result['families'], 2)
        self.assertEqual(result['success_rate_family_bootstrap_ci95'], [0., 1.])
        self.assertEqual(result, evaluation.aggregate(rows))
        self.assertEqual(evaluation.aggregate([]), {})

    def test_first_n_and_macro_mean_cover_only_observed_categories(self):
        rows = [self.row('a'), self.row('b', event='ordinary', stop_at=1), self.row('jump')]
        rows[2]['category'] = 'jump'
        clips, summary, _, _ = self.run_evaluation(rows, max_motions=2)
        self.assertEqual(len(clips), 12)
        self.assertEqual({c['motion_id'] for c in clips}, {'a', 'b'})
        for metrics in summary['matrix'].values():
            self.assertEqual(set(metrics['categories']), {'main'})
            self.assertEqual(metrics['macro_category_success'], .5)
        _, full, _, _ = self.run_evaluation(rows, max_motions=99)
        for metrics in full['matrix'].values():
            self.assertEqual(set(metrics['categories']), {'main', 'jump'})
            self.assertAlmostEqual(metrics['success_rate'], 2 / 3)
            self.assertEqual(metrics['macro_category_success'], .75)

    def test_real_cpu_bank_preserves_shard_and_label_order(self):
        root = self.root / 'synthetic_bank'
        root.mkdir()
        shards = []
        rows = [self.row('main_b'), self.row('main_a'), self.row('jump_a')]
        rows[-1]['category'] = 'jump'
        for i, labels in enumerate((rows[:2], rows[2:])):
            folder = root / f'shard_{i}'
            folder.mkdir()
            frames = len(labels) * 4
            shards.append(dict(path=folder.name, starts=list(range(0, frames, 4)),
                               ends=list(range(4, frames + 1, 4)), frames=frames, count=len(labels)))
            (folder / 'motions.json').write_text(json.dumps(labels))
            for source in SOURCES:
                np.save(folder / f'{source}_command.npy', np.zeros((frames, 1), dtype=np.float16))
        (root / 'dataset.json').write_text(json.dumps({'splits': {'val': shards}, 'offsets': [0]}))
        (root / 'READY').write_text(sha256(root / 'dataset.json'))
        bank = CommandBank(root, 'val', 'cpu')
        self.assertEqual([r['motion_id'] for r in bank.rows], ['main_b', 'main_a', 'jump_a'])
        self.assertEqual([r['category'] for r in bank.rows[:2]], ['main', 'main'])
        self.assertEqual(bank.starts.tolist(), [0, 4, 8])
        self.assertEqual(bank.lengths.tolist(), [4, 4, 4])

    def metric_fixture(self, *, actual_root, actual_point, actual_quat=(1., 0., 0., 0.),
                       actual_velocity=(0., 0., 0.)):
        origin = torch.tensor([[10., -7., 0.]])
        ref = SimpleNamespace(root_pos_w=torch.tensor([[[0., 0., 1.]]]),
                              root_quat_w=torch.tensor([[[1., 0., 0., 0.]]]),
                              body_pos_w=torch.tensor([[[[1., 0., 1.]]]]),
                              body_vel_w=torch.tensor([[[[1., 0., 0.]]]]))
        data = SimpleNamespace(root_link_pos_w=torch.tensor([actual_root]) + origin,
                               body_link_pos_w=torch.tensor([[actual_point]]) + origin[:, None],
                               root_link_quat_w=torch.tensor([actual_quat]),
                               body_link_lin_vel_w=torch.tensor([[actual_velocity]]))
        return SimpleNamespace(command=SimpleNamespace(_motion=ref, keypoint_idx_asset=[0],
                                                        keypoint_idx_motion=[0]),
                               robot=SimpleNamespace(data=data),
                               env=SimpleNamespace(scene=SimpleNamespace(env_origins=origin)))

    def test_root_relative_removes_translation_and_environment_origin(self):
        pool = self.metric_fixture(actual_root=(3., 4., 1.), actual_point=(4., 4., 1.))
        metrics = XWBCPool.metrics(pool)
        self.assertEqual(set(metrics), set(ERROR_KEYS))
        self.assertEqual(metrics['mpkpe_world_m'].item(), 5.)
        self.assertEqual(metrics['root_position_m'].item(), 5.)
        self.assertEqual(metrics['mpkpe_root_relative_m'].item(), 0.)
        offset = torch.tensor([[20., 30., 0.]])
        pool.env.scene.env_origins += offset
        pool.robot.data.root_link_pos_w += offset
        pool.robot.data.body_link_pos_w += offset[:, None]
        for key, value in XWBCPool.metrics(pool).items():
            torch.testing.assert_close(value, metrics[key])

    def test_root_relative_uses_world_axes_and_velocity_is_world_velocity(self):
        half = 2 ** -.5
        pool = self.metric_fixture(actual_root=(0., 0., 1.), actual_point=(0., 1., 1.),
                                   actual_quat=(half, 0., 0., half), actual_velocity=(0., 1., 0.))
        metrics = XWBCPool.metrics(pool)
        for key in ('mpkpe_root_relative_m', 'mpkpe_world_m', 'velocity_mps'):
            self.assertAlmostEqual(metrics[key].item(), 2 ** .5, places=6)
        self.assertAlmostEqual(metrics['root_rotation_rad'].item(), np.pi / 2, places=6)
        self.assertEqual(metrics['root_position_m'].item(), 0.)
        pool.robot.data.root_link_quat_w *= -1
        for key, value in XWBCPool.metrics(pool).items():
            torch.testing.assert_close(value, metrics[key])


if __name__ == '__main__':
    unittest.main()
