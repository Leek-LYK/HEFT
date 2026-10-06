"""Prespecified A_T -> B_T -> E_T pipeline with unchanged training budgets and full gates."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
from run_xwbc_no_c_ablation import ROOT,save,sha
from finish_xwbc_no_c_evaluation import audit_scores
from compare_xwbc_refinement import compare


def call(command,log):
 with log.open('x') as f:subprocess.run([sys.executable,'-u','-B',*command],stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();os.chdir(ROOT)
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 config='cfg/xwbc/v02/staged_transformer.yaml';ecfg='cfg/xwbc/v02/refinement_transformer.yaml'
 ref=ROOT/'logs/v0.2/no_c_ablation_20261005/direct_e_run01';refck=ref/'checkpoint_2300.pt'
 save(out/'protocol.json',dict(route='A_T->B_T->E_T',created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
      A_T_updates_per_teacher=10000,A_T_transitions_per_teacher=1310720000,B_T_updates=2000,B_T_transitions=65536000,
      E_T_updates=2300,E_T_transitions=75366400,student_optimizer_steps=132800,student_replay_samples=534118400,
      selection='fixed_final_endpoints_before_training',reference_checkpoint_sha256=sha(refck),
      reference_summaries={s:sha(ref/f'eval_{s}_2300/summary.json') for s in ['val','test']},
      configs={c:sha(ROOT/c) for c in [config,ecfg]}))
 workers={}
 try:
  for robot in ['g1','roban']:
   with (out/f'A_T_{robot}.driver.log').open('x') as log:
    workers[robot]=subprocess.Popen([sys.executable,'-u','-B','scripts/xwbc_staged.py','teacher','--robot',robot,'--config',config,
                                   '--output',str(out/f'teacher_{robot}')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  save(out/'A_T.launch.json',{r:p.pid for r,p in workers.items()})
  for r,p in workers.items():
   if p.wait()!=0:raise RuntimeError('A_T training failed: '+r)
  for r in ['g1','roban']:
   run=out/f'teacher_{r}'
   call(['scripts/audit_xwbc_transformer.py','--run',str(run)],out/f'A_T_{r}.audit.log')
   call(['scripts/xwbc_staged.py','evaluate','--checkpoint',str(run/'checkpoint_final.pt'),'--output',str(run/'eval_val'),'--num-envs','64'],out/f'A_T_{r}.eval.log')
   gate=json.loads((run/'eval_val/gate.json').read_text())
   if not gate['passed']:raise RuntimeError('A_T admission failed: '+r+' '+str(gate['issues']))
  b=out/'student_human';command=['scripts/xwbc_staged.py','student','--config',config,'--output',str(b)]
  for r in ['g1','roban']:command += ['--'+r+'-teacher',str(out/f'teacher_{r}/checkpoint_final.pt'),'--'+r+'-gate',str(out/f'teacher_{r}/eval_val/gate.json')]
  call(command,out/'B_T.driver.log');call(['scripts/audit_xwbc_transformer.py','--run',str(b)],out/'B_T.audit.log')
  call(['scripts/xwbc_staged.py','evaluate','--checkpoint',str(b/'checkpoint_final.pt'),'--output',str(b/'eval_val'),'--num-envs','64'],out/'B_T.eval.log')
  gate=json.loads((b/'eval_val/gate.json').read_text())
  if not gate['passed']:raise RuntimeError('B_T admission failed: '+str(gate['issues']))
  e=out/'E_T'
  call(['scripts/refine_xwbc_student.py','--config',ecfg,'--baseline-root',str(out),'--output',str(e)],out/'E_T.driver.log')
  call(['scripts/audit_xwbc_transformer.py','--run',str(e)],out/'E_T.audit.log')
  call(['scripts/audit_xwbc_refinement.py','--baseline',str(b/'checkpoint_final.pt'),'--run',str(e),'--checkpoint',str(e/'checkpoint_2300.pt')],out/'E_T.frozen.audit.log')
  save(e/'selection.json',dict(endpoint=2300,criterion='fixed_before_training',sha256=sha(e/'checkpoint_2300.pt')))
  for split,nenv in [('val',64),('test',500)]:
   dest=e/f'eval_{split}_2300';baseline=ref/f'eval_{split}_2300'
   call(['scripts/xwbc_staged.py','evaluate','--checkpoint',str(e/'checkpoint_2300.pt'),'--output',str(dest),'--split',split,'--num-envs',str(nenv),'--seed','42'],out/f'E_T.eval_{split}.log')
   for r in ['g1','roban']:assert json.loads((dest/f'{r}_environment.json').read_text())==json.loads((baseline/f'{r}_environment.json').read_text())
   registration=json.loads((out/'protocol.json').read_text());assert sha(baseline/'summary.json')==registration['reference_summaries'][split]
   save(dest/'independent_score_audit.json',dict(candidate=audit_scores(dest,e/'checkpoint_2300.pt'),reference=audit_scores(baseline,refck)))
   save(e/f'comparison_{split}.json',compare(baseline,dest))
  save(out/'pipeline_result.json',dict(status='complete',route='A_T->B_T->E_T',test_used_for_selection=False))
 except Exception as exc:
  save(out/'pipeline_failure.json',dict(status='stopped',reason=str(exc),other_workers_preserved=True));raise


if __name__=='__main__':main()
