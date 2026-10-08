"""A10000->A15000 and two matched E2300->E+1000 eight-step student arms."""
import argparse,datetime,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from scripts.run_xwbc_no_c_ablation import save,sha
from scripts.compare_xwbc_refinement import compare
from active_adaptation.learning.xwbc_staged import verify_gate
from active_adaptation.learning.xwbc import load_train_config
from active_adaptation.learning.modules.xwbc_incremental import load_checkpoint_safe,eight_step_origin,equal_tree

OLD=ROOT/'logs/v0.2/staged_run01'
FLOW=ROOT/'logs/v0.2/action_decoder_20261007/formal/flow/E_FLOW'
FSQ=ROOT/'logs/v0.2/fsq_20261007/run01/E_FSQ'
TCFG='cfg/xwbc/v02/teacher_resume_15k.yaml'
SCFG='cfg/xwbc/v02/refinement_flow_incremental.yaml'


def call(command,log):
    with log.open('x') as f:
        subprocess.run([sys.executable,'-u','-B',*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def audit_scores(directory,checkpoint):
    summary=json.loads((directory/'summary.json').read_text());gate=json.loads((directory/'gate.json').read_text())
    assert sha(directory/'summary.json')==gate['summary_sha256']
    assert summary['checkpoint_sha256']==sha(checkpoint) and summary['full_scope'] and summary['expected_clips']==500
    rows=[json.loads(x) for x in (directory/'clips.jsonl').read_text().splitlines()]
    assert len(rows)==500*len(summary['matrix'])==len({(x['source'],x['robot'],x['motion_id']) for x in rows})
    for route,stored in summary['matrix'].items():
        source,robot=route.split('->');rr=[x for x in rows if x['source']==source and x['robot']==robot];assert len(rr)==500
        values=dict(success_rate=sum(x['success'] for x in rr)/500,completion=sum(x['completion'] for x in rr)/500,
                    recovery_clips=sum(x['reason']=='nonfinite' for x in rr))
        for k in rr[0]['errors']:values[k]=sum(x['errors'][k]*x['steps'] for x in rr)/sum(x['steps'] for x in rr)
        assert all(math.isfinite(v) and math.isclose(v,stored[k],rel_tol=1e-10,abs_tol=1e-12) for k,v in values.items())
    return dict(passed=True,records=len(rows),checkpoint_sha256=summary['checkpoint_sha256'],summary_sha256=sha(directory/'summary.json'))


def same_environments(a,b):
    for robot in ('g1','roban'):
        if (a/f'{robot}_environment.json').exists():
            assert json.loads((a/f'{robot}_environment.json').read_text())==json.loads((b/f'{robot}_environment.json').read_text())


def teacher_worker(out,robot):
    run=out/'teachers'/f'teacher_{robot}';source=OLD/f'teacher_{robot}/checkpoint_final.pt'
    call(['scripts/xwbc_staged.py','teacher','--robot',robot,'--config',TCFG,'--updates','15000',
          '--resume',str(source),'--output',str(run)],out/f'A15_{robot}.train.log')
    call(['scripts/audit_xwbc_incremental.py','--kind','teacher','--source',str(source),'--run',str(run)],out/f'A15_{robot}.audit.log')
    ck=run/'checkpoint_final.pt';dest=run/'eval_val'
    call(['scripts/xwbc_staged.py','evaluate','--checkpoint',str(ck),'--output',str(dest),'--num-envs','64','--seed','42'],out/f'A15_{robot}.eval.log')
    original=OLD/f'teacher_{robot}/eval_val_success_priority'
    same_environments(dest,OLD/f'teacher_{robot}/eval_val')
    save(dest/'independent_score_audit.json',dict(candidate=audit_scores(dest,ck),reference=audit_scores(original,source)))
    save(run/'comparison_to_A10000.json',compare(original,dest))
    verify_gate(dest/'gate.json',ck,json.loads((run/'config.json').read_text()),'A',robot)
    save(run/'pipeline_result.json',dict(status='complete',update=15000,admitted=True,improvement_not_assumed=True))


def student_worker(out,arm):
    run=out/arm;teacher_root=OLD if arm=='A10000_control' else out/'teachers'
    expected=10000 if arm=='A10000_control' else 15000
    origin=out/'origin8';checkpoint=origin/'checkpoint_2300.pt'
    command=['scripts/refine_xwbc_student.py','--config',SCFG,'--output',str(run),
             '--student-checkpoint',str(checkpoint),'--student-validation',str(origin/'eval_val'),
             '--expected-teacher-update',str(expected)]
    for robot in ('g1','roban'):
        td=teacher_root/f'teacher_{robot}'
        gate=td/('eval_val_success_priority' if expected==10000 else 'eval_val')/'gate.json'
        command += ['--'+robot+'-teacher',str(td/'checkpoint_final.pt'),'--'+robot+'-gate',str(gate)]
    call(command,out/f'{arm}.train.log')
    call(['scripts/audit_xwbc_incremental.py','--kind','student','--source',str(checkpoint),'--run',str(run)],out/f'{arm}.audit.log')
    ck=run/'checkpoint_1000.pt'
    # Audit binds final checkpoint tensors; verify the selected saved endpoint as well.
    endpoint=load_checkpoint_safe(ck)
    final=load_checkpoint_safe(run/'checkpoint_final.pt')
    assert equal_tree(endpoint['model'],final['model']) and endpoint['update']==1000
    save(run/'selection.json',dict(update=1000,sha256=sha(ck),criterion='fixed_before_training',flow_steps=8))
    for split,nenv in [('val',64),('test',500)]:
        dest=run/f'eval_{split}_1000'
        call(['scripts/xwbc_staged.py','evaluate','--checkpoint',str(ck),'--output',str(dest),'--split',split,
              '--num-envs',str(nenv),'--seed','42'],out/f'{arm}.{split}.log')
        baseline=origin/f'eval_{split}'
        # The origin evaluator is independent and must finish before comparison.
        while not (baseline/'independent_score_audit.json').exists():
            failure=out/f'origin_{split}.failure.json'
            if failure.exists():raise RuntimeError('Origin evaluation failed: '+split)
            time.sleep(15)
        same_environments(dest,baseline)
        save(dest/'independent_score_audit.json',audit_scores(dest,ck))
        save(run/f'comparison_to_origin_{split}.json',compare(baseline,dest))
        fsq=FSQ/f'eval_{split}_2300';same_environments(dest,fsq)
        audit_scores(fsq,FSQ/'checkpoint_2300.pt')
        save(run/f'comparison_to_FSQ_{split}.json',compare(fsq,dest))
    save(run/'pipeline_result.json',dict(status='complete',teacher_updates=expected,student_additional_updates=1000,flow_steps=8,test_used_for_selection=False))


def pipeline(out):
    out.mkdir(parents=True,exist_ok=False);(out/'teachers').mkdir();origin=out/'origin8';origin.mkdir()
    source=FLOW/'checkpoint_2300.pt';source_cfg=json.loads((FLOW/'config.json').read_text())
    verify_gate(FLOW/'eval_val_steps_8/gate.json',source,source_cfg,'E')
    assert json.loads((FLOW/'eval_val_steps_8/summary.json').read_text())['inference']['steps']==8
    teachers={}
    for robot in ('g1','roban'):
        td=OLD/f'teacher_{robot}';ck=td/'checkpoint_final.pt'
        approved=dict(json.loads((td/'config.json').read_text()),acceptance=load_train_config(ROOT/TCFG)['acceptance'])
        verify_gate(td/'eval_val_success_priority/gate.json',ck,approved,'A',robot)
        teachers[robot]=dict(checkpoint=str(ck),sha256=sha(ck))
    state=load_checkpoint_safe(source)
    converted=eight_step_origin(state,source,sha(source));ck=origin/'checkpoint_2300.pt'
    torch.save(converted,ck);save(origin/'conversion.json',converted['solver_conversion']);save(origin/'config.json',converted['config'])
    protocol=dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        experiment='A15k and matched eight-step E2300 incremental refinement',teachers=teachers,
        source_student=str(source),source_student_sha256=sha(source),origin8_sha256=sha(ck),
        teacher_per_robot=dict(start=10000,endpoint=15000,additional_transitions=655360000,additional_optimizer_steps=240000),
        student_per_arm=dict(updates=1000,transitions=32768000,optimizer_steps=16000,replay_samples=32768000),
        arms=['A10000_control','A15000_candidate'],flow_steps=8,anchor='frozen starting E2300 at eight steps',
        optimizer_and_replay='new in both student arms',test_used_for_selection=False,
        prior_test_already_seen=True,teacher_admission='existing absolute capability thresholds; not assumed better than A10000',
        configs={p:sha(ROOT/p) for p in [TCFG,SCFG]})
    save(out/'protocol.json',protocol)
    files=[TCFG,SCFG,'cfg/xwbc/v02/staged.yaml','cfg/xwbc/v02/source_adaptation.yaml',
           'scripts/run_xwbc_incremental.py','scripts/audit_xwbc_incremental.py','scripts/refine_xwbc_student.py',
           'scripts/compare_xwbc_refinement.py','scripts/run_xwbc_no_c_ablation.py',
           'active_adaptation/learning/xwbc_staged.py','active_adaptation/learning/xwbc.py',
           'active_adaptation/learning/modules/xwbc.py','active_adaptation/learning/modules/xwbc_staged.py',
           'active_adaptation/learning/modules/xwbc_action_decoder.py','active_adaptation/learning/modules/xwbc_incremental.py']
    for name in files:
        dest=out/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    save(out/'source_hashes.json',{name:sha(ROOT/name) for name in files})
    del state,converted
    workers={};finished={};launched=set()
    def spawn(name,command):
        with (out/f'{name}.driver.log').open('x') as log:
            workers[name]=subprocess.Popen([sys.executable,'-u','-B',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        launched.add(name)
        (out/'workers.json').write_text(json.dumps({n:p.pid for n,p in workers.items()},indent=2))
    for robot in ('g1','roban'):
        spawn('teacher_'+robot,['scripts/run_xwbc_incremental.py','--mode','teacher','--robot',robot,'--output',str(out)])
    for split,nenv in [('val',64),('test',500)]:
        spawn('origin_'+split,['scripts/xwbc_staged.py','evaluate','--checkpoint',str(ck),'--output',str(origin/f'eval_{split}'),
              '--split',split,'--num-envs',str(nenv),'--seed','42'])
    while True:
        for name,p in list(workers.items()):
            if name in finished or p.poll() is None:continue
            finished[name]=p.returncode
            if name.startswith('origin_'):
                split=name.removeprefix('origin_')
                try:
                    if p.returncode:raise RuntimeError('Evaluation process failed')
                    dest=origin/f'eval_{split}';audit=audit_scores(dest,ck)
                    if split=='val':verify_gate(dest/'gate.json',ck,json.loads((origin/'config.json').read_text()),'E')
                    save(dest/'independent_score_audit.json',audit)
                except Exception as exc:
                    finished[name]=1;save(out/f'{name}.failure.json',dict(reason=str(exc)))
            save(out/f'{name}.exit.json',dict(exit_code=finished[name]))
        origin_ready=finished.get('origin_val')==0
        origin_bad=any(finished.get('origin_'+s,0)!=0 for s in ('val','test'))
        if origin_ready and not origin_bad and 'A10000_control' not in launched:
            spawn('A10000_control',['scripts/run_xwbc_incremental.py','--mode','student','--arm','A10000_control','--output',str(out)])
        teachers_done=all('teacher_'+r in finished for r in ('g1','roban'))
        teachers_ok=teachers_done and all(finished['teacher_'+r]==0 for r in ('g1','roban'))
        if origin_ready and not origin_bad and teachers_ok and 'A15000_candidate' not in launched:
            spawn('A15000_candidate',['scripts/run_xwbc_incremental.py','--mode','student','--arm','A15000_candidate','--output',str(out)])
        all_idle=all(name in finished for name in workers)
        if all_idle:break
        time.sleep(15)
    if all(finished.get(a)==0 for a in ('A10000_control','A15000_candidate')):
        initial=[torch.load(out/a/'initial_model.pt',map_location='cpu',weights_only=True) for a in ('A10000_control','A15000_candidate')]
        assert equal_tree(*initial)
        save(out/'initialization_audit.json',dict(passed=True,initial_model_tensors='bit_identical',origin_sha256=sha(ck)))
        for split in ('val','test'):
            save(out/f'A15000_vs_A10000_{split}.json',compare(out/'A10000_control'/f'eval_{split}_1000',out/'A15000_candidate'/f'eval_{split}_1000'))
    complete=all(finished.get(n)==0 for n in ('teacher_g1','teacher_roban','origin_val','origin_test','A10000_control','A15000_candidate'))
    save(out/'pipeline_result.json',dict(status='complete' if complete else 'finished_with_stopped_arm',workers=finished,test_used_for_selection=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['pipeline','teacher','student'],default='pipeline')
    p.add_argument('--robot',choices=['g1','roban']);p.add_argument('--arm',choices=['A10000_control','A15000_candidate'])
    args=p.parse_args();os.chdir(ROOT);torch.set_num_threads(1);out=args.output.resolve()
    try:
        if args.mode=='pipeline':pipeline(out)
        elif args.mode=='teacher':teacher_worker(out,args.robot)
        else:student_worker(out,args.arm)
    except Exception as exc:
        if out.exists():
            name='pipeline' if args.mode=='pipeline' else ('teacher_'+args.robot if args.mode=='teacher' else args.arm)
            with (out/f'{name}.failure.json').open('x') as f:json.dump(dict(reason=str(exc)),f,indent=2)
        raise
if __name__=='__main__':main()
