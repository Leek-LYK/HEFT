"""Finish a preregistered no-C endpoint with concurrent validation/test and raw-score audits."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from run_xwbc_no_c_ablation import ROOT, checked_reference, save, sha
from compare_xwbc_refinement import compare


def audit_scores(directory, checkpoint):
    summary=checked_reference(directory,checkpoint)
    rows=[json.loads(x) for x in (directory/'clips.jsonl').read_text().splitlines()]
    assert len(rows)==3000
    assert len({(x['source'],x['robot'],x['motion_id']) for x in rows})==3000
    for route,stored in summary['matrix'].items():
        source,robot=route.split('->')
        rr=[x for x in rows if (x['source'],x['robot'])==(source,robot)]
        assert len(rr)==500
        assert len({x['family_id'] for x in rr})==stored['families']
        actual=dict(clips=len(rr),success_rate=sum(x['success'] for x in rr)/len(rr),
                    completion=sum(x['completion'] for x in rr)/len(rr),
                    recovery_clips=sum(x['reason']=='nonfinite' for x in rr))
        for key in rr[0]['errors']:
            actual[key]=sum(x['errors'][key]*x['steps'] for x in rr)/max(1,sum(x['steps'] for x in rr))
        for key,value in actual.items():
            assert math.isfinite(value) and math.isclose(value,stored[key],rel_tol=1e-10,abs_tol=1e-12),(route,key)
    return dict(passed=True,records=len(rows),checkpoint_sha256=summary['checkpoint_sha256'],
                summary_sha256=sha(directory/'summary.json'),raw_scores='independently_recomputed')


def alive(pid):
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]!='Z'
    except (ProcessLookupError,FileNotFoundError):
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--validation-pid',type=int,required=True)
    args=p.parse_args();os.chdir(ROOT);run=args.run.resolve()
    protocol=json.loads((run.parent/(run.name+'.protocol.json')).read_text())
    selected=json.loads((run/'selection.json').read_text())
    checkpoint=run/'checkpoint_2300.pt';reference=Path(protocol['reference_checkpoint'])
    assert sha(checkpoint)==selected['sha256'] and selected['iteration']==protocol['endpoint']==2300
    assert sha(reference)==protocol['reference_sha256']
    subprocess.run([sys.executable,'-B','scripts/audit_xwbc_refinement.py','--baseline',
                    'logs/v0.2/staged_run01/student_human/checkpoint_final.pt','--run',str(run),
                    '--checkpoint',str(checkpoint),'--output',str(run/'checkpoint_2300.audit.json')],check=True)
    test=run/'eval_test_2300'
    with (run/'eval_test.console.log').open('x') as log:
        worker=subprocess.Popen([sys.executable,'-u','-B','scripts/xwbc_staged.py','evaluate',
                                 '--checkpoint',str(checkpoint),'--output',str(test),'--num-envs','500',
                                 '--split','test','--seed','42'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    pids=dict(val=args.validation_pid,test=worker.pid)
    save(run/'parallel_evaluation.launch.json',dict(pids=pids,started_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
    pending=set(pids);audits={}
    while pending:
        for split in sorted(pending.copy()):
            directory=run/f'eval_{split}_2300'
            if not (directory/'gate.json').exists():
                if not alive(pids[split]):raise RuntimeError(f'{split} stopped before completing evaluation')
                continue
            baseline=Path(protocol['references'][split]['directory'])
            assert sha(baseline/'summary.json')==protocol['references'][split]['summary_sha256']
            for robot in ('g1','roban'):
                assert json.loads((directory/f'{robot}_environment.json').read_text())==json.loads((baseline/f'{robot}_environment.json').read_text())
            audits[split]=dict(candidate=audit_scores(directory,checkpoint),reference=audit_scores(baseline,reference),environments='identical')
            save(directory/'independent_score_audit.json',audits[split])
            result=compare(baseline,directory)
            save(run/f'comparison_{split}.json',result)
            print(json.dumps(dict(split=split,means=result['means'],success_ci=result['pooled_success_delta_ci95_pp'],other_ci=result['mean_delta_family_ci95'])),flush=True)
            pending.remove(split)
        if pending:time.sleep(15)
    if worker.wait()!=0:raise RuntimeError('Test evaluator failed')
    save(run/'final_evaluation_audit.json',dict(passed=True,endpoint_sha256=sha(checkpoint),reference_sha256=sha(reference),splits=audits,
                                              fixed_endpoint=True,protocol_sha256=sha(run.parent/(run.name+'.protocol.json')),
                                              selection_sha256=sha(run/'selection.json'),analysis_script_sha256=sha(Path(__file__))))
    save(run/'pipeline_result.json',dict(status='complete',endpoint=2300,reference='A->B->C->E all_sources checkpoint_300',
                                        candidate='A->B->E checkpoint_2300',test_used_for_selection=False))


if __name__=='__main__':main()
