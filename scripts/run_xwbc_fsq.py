"""Reuse A, train B/E with command FSQ, then compare fixed endpoints to best A/B/E."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_xwbc_no_c_ablation import save, sha, checked_reference
from scripts.finish_xwbc_no_c_evaluation import audit_scores
from scripts.compare_xwbc_refinement import compare
from active_adaptation.learning.xwbc import load_train_config
from active_adaptation.learning.xwbc_staged import gate_decision, verify_gate


def call(command, log):
    with log.open('x') as stream:
        subprocess.run([sys.executable, '-u', '-B', *command], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def capability_gate(run, cfg):
    """Apply the baseline's already established B admission criteria to full scores."""
    original = run/'eval_val'
    summary = json.loads((original/'summary.json').read_text())
    old_gate = json.loads((original/'gate.json').read_text())
    assert sha(original/'summary.json') == old_gate['summary_sha256']
    revised = copy.deepcopy(summary)
    revised['acceptance'] = cfg['acceptance']
    revised['reassessment'] = dict(source_summary_sha256=sha(original/'summary.json'),
        source_clips_sha256=sha(original/'clips.jsonl'), original_acceptance=summary['acceptance'],
        original_gate=old_gate, new_rollouts=False,
        reason='Prespecified same capability admission as baseline B->E')
    output = run/'eval_val_capability'
    output.mkdir()
    save(output/'summary.json', revised)
    save(output/'gate.json', dict(**gate_decision(revised), summary_sha256=sha(output/'summary.json')))
    (output/'clips.jsonl').symlink_to('../eval_val/clips.jsonl')
    return verify_gate(output/'gate.json', run/'checkpoint_final.pt', cfg, 'B')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default='cfg/xwbc/v02/staged_fsq.yaml')
    parser.add_argument('--refinement-config', default='cfg/xwbc/v02/refinement_fsq.yaml')
    parser.add_argument('--reference', type=Path, default=ROOT/'logs/v0.2/no_c_ablation_20261005/direct_e_run01')
    parser.add_argument('--student-reference', type=Path, default=ROOT/'logs/v0.2/staged_run01/student_human/eval_val')
    parser.add_argument('--label', default='FSQ')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config, ecfg, label = args.config, args.refinement_config, args.label
    original = ROOT/'logs/v0.2/staged_run01'
    ref = args.reference.resolve()
    refck = ref/'checkpoint_2300.pt'
    cfg = load_train_config(ROOT/config)
    cfg['dataset'] = str((ROOT/cfg['dataset']).resolve())
    e_cfg = load_train_config(ROOT/ecfg)
    e_cfg['dataset'] = cfg['dataset']
    teachers = {}
    for robot in ('g1', 'roban'):
        checkpoint = original/f'teacher_{robot}/checkpoint_final.pt'
        gate = original/f'teacher_{robot}/eval_val_success_priority/gate.json'
        verify_gate(gate, checkpoint, cfg, 'A', robot)
        teachers[robot] = dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), gate=str(gate))
        (out/f'teacher_{robot}').symlink_to(original/f'teacher_{robot}', target_is_directory=True)
    for split in ('val', 'test'):
        checked_reference(ref/f'eval_{split}_2300', refck)
    protocol = dict(route=f'A(existing)->B_{label}->E_{label}',
        created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        student_fsq=cfg['student_fsq'], action_decoder=cfg.get('action_decoder'),
        scope='FSQ command encoders and optional robot-conditioned action decoder',
        B_updates=2000, B_transitions=65536000, E_updates=2300, E_transitions=75366400,
        optimizer_steps=132800, replay_samples=534118400, seed=0,
        endpoint_selection='fixed before training; no test-based selection',
        B_admission=e_cfg['acceptance'], E_performance_gate_before_test=False,
        teachers=teachers, reference_checkpoint=str(refck), reference_sha256=sha(refck),
        reference_summaries={s:sha(ref/f'eval_{s}_2300/summary.json') for s in ('val','test')},
        configs={c:sha(ROOT/c) for c in (config, ecfg)})
    save(out/'protocol.json', protocol)
    try:
        b = out/'student_human'
        command = ['scripts/xwbc_staged.py', 'student', '--config', config, '--output', str(b)]
        for r, rec in teachers.items():
            command += ['--'+r+'-teacher', rec['checkpoint'], '--'+r+'-gate', rec['gate']]
        call(command, out/f'B_{label}.driver.log')
        call(['scripts/audit_xwbc_fsq.py', '--run', str(b)], out/f'B_{label}.audit.log')
        call(['scripts/xwbc_staged.py', 'evaluate', '--checkpoint', str(b/'checkpoint_final.pt'),
              '--output', str(b/'eval_val'), '--num-envs', '64', '--seed', '42'], out/f'B_{label}.eval.log')
        save(b/'comparison_val.json', compare(args.student_reference, b/'eval_val'))
        if cfg.get('action_decoder'):
            call(['scripts/audit_xwbc_action_decoder.py','--run',str(b)],out/f'B_{label}.decoder.audit.log')
        capability_gate(b, e_cfg)
        e = out/f'E_{label}'
        call(['scripts/refine_xwbc_student.py', '--config', ecfg, '--baseline-root', str(out),
              '--output', str(e)], out/f'E_{label}.driver.log')
        call(['scripts/audit_xwbc_fsq.py', '--run', str(e)], out/f'E_{label}.audit.log')
        if cfg.get('action_decoder'):
            call(['scripts/audit_xwbc_action_decoder.py','--run',str(e)],out/f'E_{label}.decoder.audit.log')
        ck = e/'checkpoint_2300.pt'
        call(['scripts/audit_xwbc_refinement.py', '--baseline', str(b/'checkpoint_final.pt'),
              '--run', str(e), '--checkpoint', str(ck)], out/f'E_{label}.frozen.audit.log')
        save(e/'selection.json', dict(endpoint=2300, criterion='fixed_before_training', sha256=sha(ck)))
        for split, nenv in [('val', 64), ('test', 500)]:
            dest, baseline = e/f'eval_{split}_2300', ref/f'eval_{split}_2300'
            call(['scripts/xwbc_staged.py', 'evaluate', '--checkpoint', str(ck), '--output', str(dest),
                  '--split', split, '--num-envs', str(nenv), '--seed', '42'], out/f'E_{label}.eval_{split}.log')
            assert sha(baseline/'summary.json') == protocol['reference_summaries'][split]
            assert sha(refck) == protocol['reference_sha256']
            for robot in ('g1', 'roban'):
                assert json.loads((dest/f'{robot}_environment.json').read_text()) == json.loads((baseline/f'{robot}_environment.json').read_text())
            save(dest/'independent_score_audit.json', dict(candidate=audit_scores(dest, ck), reference=audit_scores(baseline, refck)))
            save(e/f'comparison_{split}.json', compare(baseline, dest))
        if cfg.get('action_decoder', {}).get('kind') == 'flow':
            # Secondary validation-only solver diagnostics; primary always uses 4 steps.
            for steps in (1, 2, 8):
                dest = e/f'eval_val_steps_{steps}'
                call(['scripts/xwbc_staged.py','evaluate','--checkpoint',str(ck),'--output',str(dest),
                      '--split','val','--num-envs','64','--seed','42','--flow-steps',str(steps)],
                     out/f'E_{label}.steps_{steps}.log')
                save(dest/'independent_score_audit.json', audit_scores(dest, ck))
                save(e/f'comparison_val_steps_{steps}.json', compare(ref/'eval_val_2300', dest))
        save(out/'pipeline_result.json', dict(status='complete', route=protocol['route'], test_used_for_selection=False))
    except Exception as exc:
        save(out/'pipeline_failure.json', dict(status='stopped', reason=str(exc)))
        raise


if __name__ == '__main__':
    main()
