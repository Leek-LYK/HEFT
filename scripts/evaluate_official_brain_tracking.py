"""Evaluate official ONNX with unchanged 0915 control and root-local metrics."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from analyze_brain_root_relative_tracking import ROOT, BASE, BODIES, INIT, SAMPLE

POLICY = ROOT / 'dataset/_audit/roban_ominisoma/官方heft/policy0905.onnx'
OUTPUT = BASE / 'official_policy0905_tracking_20260919'
FINISH = '''
pos = np.asarray(relative_positions).reshape(-1, len(metric_names))
ori = np.asarray(relative_orientations).reshape(-1, len(metric_names))
assert np.isfinite(pos).all() and np.isfinite(ori).all()
assert np.allclose(csv[:, :2], 0.0)
success = report['playback_started'] and report['playback_complete'] and not report['fell']
if success:
 assert len(pos) == 298
np.savez(OUT / 'root_relative_samples.npz', position_mm=pos, orientation_rad=ori, frame=np.asarray(metric_frames), bodies=np.asarray(metric_names))
result = {'case': CASE_NAME, 'success': success, 'fell': report['fell'],
 'playback_started': report['playback_started'], 'playback_complete': report['playback_complete'],
 'sample_count': len(pos), 'bodies': metric_names,
 'root_relative_body_pos_mm': float(pos.mean()) if len(pos) else None,
 'root_relative_body_ori_rad': float(ori.mean()) if len(ori) else None,
 'joint_rmse_rad': report.get('tracking_errors', {}).get('joint_rmse_rad'),
 'source_simulate_sha256': SOURCE_HASH, 'source_csv_sha256': report['source_csv_sha256'],
 'policy_sha256': POLICY_HASH}
(OUT / 'relative_metrics.json').write_text(json.dumps(result, indent=2) + '\\n')
'''

def evaluate(case):
    out = OUTPUT / case
    out.mkdir(parents=True, exist_ok=True)
    source = (BASE / case / 'simulate.py').read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    start = source.index("sys.path.insert(0,str(ROOT/")
    end = source.index("csvpath=", start)
    source = source[:start] + (
        f"official_policy=Path({str(POLICY)!r})\n"
        "modelpaths.append(str(official_policy));sessions.append(ort.InferenceSession(str(official_policy),sess_options=so,providers=['CPUExecutionProvider']))\n"
    ) + source[end:]
    for old, new in {
        'OUT=Path(__file__).parent': f'OUT=Path({str(out)!r})',
        'tracking=[];trace=[];video_origin=np.zeros(2);fallen=False': INIT + '\ntracking=[];trace=[];video_origin=np.zeros(2);fallen=False',
        '   tracking.append(': SAMPLE + '   tracking.append(',
    }.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    prefix = f'BODY_NAMES={BODIES!r}\nCASE_NAME={case!r}\nSOURCE_HASH={source_hash!r}\nPOLICY_HASH={hashlib.sha256(POLICY.read_bytes()).hexdigest()!r}\n'
    script = out / 'evaluate_instrumented.py'
    script.write_text(prefix + source + '\n' + FINISH)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (out / 'evaluation.log').open('w') as log:
        subprocess.run([sys.executable, '-B', str(script)], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / 'relative_metrics.json').read_text())
    print(case + ': ' + ('complete' if result['success'] else 'not complete'), flush=True)
    return result

if __name__ == '__main__':
    cases = [p.parent.name for p in sorted(BASE.glob('*/report.json'))]
    assert len(cases) == 45
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate, cases))
    (OUTPUT / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
