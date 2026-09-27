"""Replay stored successful cases without rendering; instrument FK on separate data.

Original simulation inputs/reports are never overwritten. Run with the project
Python and --case <case-directory-name>, or without --case for all successes.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dataset/_audit/roban_ominisoma/brain_amp_heft_20260915_key'
OUTPUT = BASE / 'root_relative_tracking_20260919'
BODIES = ['waist_yaw_link', 'zhead_2_link', 'l_handball', 'r_handball',
          'zarm_l3_link', 'zarm_r3_link', 'zarm_l4_link', 'zarm_r4_link',
          'leg_l1_link', 'leg_r1_link', 'leg_l4_link', 'leg_r4_link',
          'leg_l6_link', 'leg_r6_link']

INIT = '''
metric_actual = mujoco.MjData(model)
metric_reference = mujoco.MjData(model)
metric_names = BODY_NAMES
metric_ids = np.array([model.body(n).id for n in metric_names])
metric_root = model.body('base_link').id
relative_positions = []
relative_orientations = []
metric_frames = []
'''
SAMPLE = '''
   # Separate FK buffers: never mutate the dynamic simulation state.
   metric_actual.qpos[:] = data.qpos
   metric_reference.qpos[:] = data.qpos
   metric_reference.qpos[:3] = csv[frame, :3]
   metric_reference.qpos[3:7] = csv[frame, 3:7]
   metric_reference.qpos[qidx] = csv[frame, 7:28]
   mujoco.mj_kinematics(model, metric_actual)
   mujoco.mj_kinematics(model, metric_reference)
   local_positions = []
   local_rotations = []
   for md in (metric_actual, metric_reference):
    rr = md.xmat[metric_root].reshape(3, 3)
    local_positions.append((md.xpos[metric_ids] - md.xpos[metric_root]) @ rr)
    local_rotations.append(np.einsum('ij,bjk->bik', rr.T, md.xmat[metric_ids].reshape(-1, 3, 3)))
   relative_positions.append(np.linalg.norm(local_positions[0] - local_positions[1], axis=1) * 1000)
   relative_orientations.append((R.from_matrix(local_rotations[1]).inv() * R.from_matrix(local_rotations[0])).magnitude())
   metric_frames.append(frame)
'''
FINISH = '''
pos = np.asarray(relative_positions)
ori = np.asarray(relative_orientations)
assert pos.shape == (298, len(metric_names)), pos.shape
assert np.isfinite(pos).all() and np.isfinite(ori).all()
assert np.allclose(csv[:, :2], 0.0)
np.savez(OUT / 'root_relative_samples.npz', position_mm=pos, orientation_rad=ori, frame=np.asarray(metric_frames), bodies=np.asarray(metric_names))
original = json.loads(ORIGINAL_REPORT.read_text())
deltas = {k: {s: abs(report['tracking_errors'][k][s] - original['tracking_errors'][k][s]) for s in ('mean', 'p95')} for k in original['tracking_errors']}
assert report['playback_complete'] and not report['fell']
assert max(v for d in deltas.values() for v in d.values()) < 1e-9, deltas
original_trace = np.load(ORIGINAL_REPORT.parent / 'trace.npy')
replay_trace = np.asarray(trace)
assert original_trace.shape == replay_trace.shape
trace_gap = float(np.max(np.abs(original_trace - replay_trace)))
assert trace_gap < 1e-9, trace_gap
result = {'case': ORIGINAL_REPORT.parent.name, 'sample_count': len(pos), 'bodies': metric_names,
 'root_relative_body_pos_mm': float(pos.mean()),
 'root_relative_body_ori_rad': float(ori.mean()),
 'frame_mean_pos_p95_mm': float(np.quantile(pos.mean(1), .95)),
 'frame_mean_ori_p95_rad': float(np.quantile(ori.mean(1), .95)),
 'joint_rmse_rad': report['tracking_errors']['joint_rmse_rad'],
 'per_body': {n: {'position_mm': float(pos[:, i].mean()), 'orientation_rad': float(ori[:, i].mean())} for i, n in enumerate(metric_names)},
 'original_metric_abs_deltas': deltas, 'trace_max_abs_delta': trace_gap,
 'source_simulate_sha256': SOURCE_HASH, 'source_csv_sha256': report['source_csv_sha256']}
(OUT / 'relative_metrics.json').write_text(json.dumps(result, indent=2) + '\\n')
'''

def replay(case):
    original = BASE / case
    out = OUTPUT / case
    out.mkdir(parents=True, exist_ok=True)
    source = (original / 'simulate.py').read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    replacements = {
        "OUT=Path(__file__).parent": f"OUT=Path({str(out)!r})",
        "tracking=[];trace=[];video_origin=np.zeros(2);fallen=False":
            INIT + "\ntracking=[];trace=[];video_origin=np.zeros(2);fallen=False",
        '   tracking.append(': SAMPLE + '   tracking.append(',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    prefix = (f'BODY_NAMES={BODIES!r}\nSOURCE_HASH={source_hash!r}\n'
              f'from pathlib import Path\nORIGINAL_REPORT=Path({str(original / "report.json")!r})\n')
    instrumented = out / 'replay_instrumented.py'
    instrumented.write_text(prefix + source + '\n' + FINISH)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (out / 'replay.log').open('w') as log:
        subprocess.run([sys.executable, '-B', str(instrumented)], cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(case + ' verified', flush=True)
    return json.loads((out / 'relative_metrics.json').read_text())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case')
    args = parser.parse_args()
    cases = []
    for p in sorted(BASE.glob('*/report.json')):
        r = json.loads(p.read_text())
        if not r['fell'] and r['playback_started'] and r['playback_complete']:
            if args.case is None or p.parent.name == args.case:
                cases.append(p.parent.name)
    assert cases
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(replay, cases))
    name = 'summary.json' if args.case is None else 'single_case_summary.json'
    (OUTPUT / name).write_text(json.dumps(results, indent=2) + '\n')
