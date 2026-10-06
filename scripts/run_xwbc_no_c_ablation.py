"""Fixed-budget B->E ablation; evaluate the prespecified endpoint even if it regresses."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_xwbc_refinement import compare


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def checked_reference(directory, checkpoint):
    summary = json.loads((directory/'summary.json').read_text())
    gate = json.loads((directory/'gate.json').read_text())
    assert sha(directory/'summary.json') == gate['summary_sha256']
    assert summary['checkpoint_sha256'] == sha(checkpoint)
    assert summary['full_scope'] and summary['expected_clips'] == 500
    assert len(summary['matrix']) == 6
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.chdir(ROOT)
    run = args.output.resolve()
    config = 'cfg/xwbc/v02/teacher_refinement_no_c.yaml'
    reference_run = ROOT/'logs/v0.2/refinement_20261005/all_sources_run01'
    reference_checkpoint = reference_run/'checkpoint_300.pt'
    references = dict(val=reference_run/'eval_val_300', test=reference_run/'eval_test_candidate')
    # Lock endpoint and reference identities before training, without selecting by scores.
    registered = dict(endpoint=2300, selection='fixed_before_training', seed=0,
                      transitions=75366400, optimizer_steps=68800, replay_samples=271974400,
                      registered_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                      reference_checkpoint=str(reference_checkpoint), reference_sha256=sha(reference_checkpoint),
                      references={})
    for split, directory in references.items():
        summary = checked_reference(directory, reference_checkpoint)
        assert summary['split'] == split
        registered['references'][split] = dict(directory=str(directory), summary_sha256=sha(directory/'summary.json'))
    save(run.parent/(run.name+'.protocol.json'), registered)
    command = [sys.executable, '-u', '-B', 'scripts/refine_xwbc_student.py', '--config', config, '--output', str(run)]
    subprocess.run(command, check=True)
    result = json.loads((run/'result.json').read_text())
    assert result['status'] == 'complete' and result['updates'] == 2300
    for key in ('optimizer_steps', 'replay_samples'):
        assert result[key] == registered[key]
    subprocess.run([sys.executable, '-B', 'scripts/audit_xwbc_refinement.py',
                    '--baseline', 'logs/v0.2/staged_run01/student_human/checkpoint_final.pt', '--run', str(run)], check=True)
    checkpoint = run/'checkpoint_2300.pt'
    save(run/'selection.json', dict(criterion='prespecified_final_endpoint_no_performance_gate',
                                   iteration=2300, checkpoint=str(checkpoint), sha256=sha(checkpoint)))
    for split, num_envs in [('val', 64), ('test', 500)]:
        output = run/('eval_'+split+'_2300')
        with (run/('eval_'+split+'.console.log')).open('x') as log:
            subprocess.run([sys.executable, '-u', '-B', 'scripts/xwbc_staged.py', 'evaluate',
                            '--checkpoint', str(checkpoint), '--output', str(output),
                            '--num-envs', str(num_envs), '--split', split, '--seed', '42'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        baseline = references[split]
        assert sha(baseline/'summary.json') == registered['references'][split]['summary_sha256']
        checked_reference(baseline, reference_checkpoint)
        checked_reference(output, checkpoint)
        for robot in ('g1', 'roban'):
            assert json.loads((output/f'{robot}_environment.json').read_text()) == json.loads((baseline/f'{robot}_environment.json').read_text()), robot
        save(run/('comparison_'+split+'.json'), compare(baseline, output))
    save(run/'pipeline_result.json', dict(status='complete', endpoint=2300,
                                         reference='A->B->C->E all_sources checkpoint_300',
                                         candidate='A->B->E checkpoint_2300', test_used_for_selection=False))


if __name__ == '__main__':
    main()
