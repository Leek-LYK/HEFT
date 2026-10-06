"""Wait for a known run, select by full validation, then test C and the frozen candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from compare_xwbc_refinement import compare


def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--driver-pid',type=int)
    p.add_argument('--evaluation-pids',help='JSON mapping checkpoint update to its live evaluation PID')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--iterations',type=int,nargs='+',default=[100,200,300])
    p.add_argument('--test-num-envs',type=int,default=64)
    p.add_argument('--baseline-test-dir',type=Path,help='Reuse an already complete, matching C test evaluation')
    args=p.parse_args()
    if args.test_num_envs<1:
        p.error('test-num-envs must be positive')
    evaluation_pids=json.loads(args.evaluation_pids) if args.evaluation_pids else {}
    if not evaluation_pids and args.driver_pid is None:
        p.error('Provide driver PID or evaluation PIDs')
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    run=args.run_dir.resolve() if args.run_dir else root/'logs/v0.2/refinement_20261005/teacher_correction_run01'
    baseline=root/'logs/v0.2/staged_run01/source_encoders/eval_val'
    candidates=[]
    for iteration in args.iterations:
        directory=run/f'eval_val_{iteration}'
        while not (directory/'gate.json').exists():
            pid=int(evaluation_pids.get(str(iteration),args.driver_pid))
            try:
                os.kill(pid,0)
                stat=Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]
                if stat=='Z':
                    raise ProcessLookupError('driver is zombie')
            except (ProcessLookupError,FileNotFoundError):
                raise RuntimeError('Driver stopped before completing required evaluation')
            time.sleep(15)
        comparison=compare(baseline,directory)
        save(run/f'comparison_val_{iteration}.json',comparison)
        print(json.dumps(dict(iteration=iteration,means=comparison['means']),ensure_ascii=False),flush=True)
        means=comparison['means']
        # Select solely on validation; do not inspect test while tuning.
        qualifies=(means['success_rate']['candidate']>means['success_rate']['baseline']
                   and means['completion']['candidate']>=means['completion']['baseline']
                   and all(means[m]['candidate']<=means[m]['baseline'] for m in ('mpkpe_root_relative_m','mpkpe_world_m'))
                   and all(r['success_delta_pp']>=-1.0-1e-8 for r in comparison['routes'].values())
                   and all(r['candidate_recoveries']<=r['baseline_recoveries'] for r in comparison['routes'].values()))
        if qualifies:
            candidates.append((means['success_rate']['candidate'],-means['mpkpe_root_relative_m']['candidate'],iteration))
    if not candidates:
        save(run/'selection.json',dict(status='no_candidate_meets_joint_validation_criteria',test_run=False))
        return
    iteration=max(candidates)[-1]
    checkpoint=run/f'checkpoint_{iteration}.pt'
    save(run/'selection.json',dict(status='selected_on_validation',iteration=iteration,checkpoint=str(checkpoint),
                                  criterion='higher mean success; no mean completion/error regression; <=1 pp route success decline; no added recoveries',
                                  test_num_envs=args.test_num_envs))
    baseline_test_output=run/'eval_test_C'
    if args.baseline_test_dir:
        baseline_test_output=args.baseline_test_dir.resolve()
        test_summary=json.loads((baseline_test_output/'summary.json').read_text())
        test_gate=json.loads((baseline_test_output/'gate.json').read_text())
        c_validation=json.loads((baseline/'summary.json').read_text())
        summary_digest=hashlib.sha256((baseline_test_output/'summary.json').read_bytes()).hexdigest()
        assert test_gate['summary_sha256']==summary_digest
        assert test_summary['split']=='test' and test_summary['full_scope']
        for key in ('checkpoint_sha256','dataset_digest','context','protocol'):
            assert test_summary[key]==c_validation[key],key
        assert set(test_summary['matrix'])==set(c_validation['matrix'])
        for robot in ('g1','roban'):
            env=json.loads((baseline_test_output/f'{robot}_environment.json').read_text())
            assert env['num_envs']==min(args.test_num_envs,test_summary['expected_clips'])
        save(run/'baseline_test_reuse.json',dict(directory=str(baseline_test_output),
             summary_sha256=summary_digest,checkpoint_sha256=test_summary['checkpoint_sha256'],
             dataset_digest=test_summary['dataset_digest'],num_envs=args.test_num_envs))
    for name,path in [('C',root/'logs/v0.2/staged_run01/source_encoders/checkpoint_final.pt'),('candidate',checkpoint)]:
        if name=='C' and args.baseline_test_dir:
            continue
        output=run/f'eval_test_{name}'
        command=[sys.executable,'-u','-B','scripts/xwbc_staged.py','evaluate','--checkpoint',str(path),
                 '--output',str(output),'--num-envs',str(args.test_num_envs),'--split','test','--seed','42']
        with (run/f'eval_test_{name}.console.log').open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    result=compare(baseline_test_output,run/'eval_test_candidate')
    save(run/'comparison_test.json',result)
    print(json.dumps(dict(test_means=result['means']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
