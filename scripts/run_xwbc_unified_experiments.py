"""Run both preregistered U variants, audit and evaluate fixed endpoints including negative results."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
from run_xwbc_no_c_ablation import ROOT,save,sha,checked_reference
from finish_xwbc_no_c_evaluation import audit_scores
from compare_xwbc_refinement import compare


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();os.chdir(ROOT)
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 ref=ROOT/'logs/v0.2/no_c_ablation_20261005/direct_e_run01';refck=ref/'checkpoint_2300.pt'
 references={s:ref/f'eval_{s}_2300' for s in ('val','test')}
 for s,d in references.items():assert checked_reference(d,refck)['split']==s
 variants=['uniform','curriculum']
 protocol=dict(stage='U',endpoint=4300,selection='fixed_before_training_for_both_variants',
               transitions_per_variant=140902400,optimizer_steps_per_variant=132800,replay_samples_per_variant=534118400,
               created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               reference_sha256=sha(refck),references={s:dict(directory=str(d),summary_sha256=sha(d/'summary.json')) for s,d in references.items()},
               config_sha256={v:sha(ROOT/f'cfg/xwbc/v02/unified_{v}.yaml') for v in variants})
 save(out/'protocol.json',protocol);workers={}
 for v in variants:
  with (out/f'{v}.driver.log').open('x') as log:
   workers[v]=subprocess.Popen([sys.executable,'-u','-B','scripts/train_xwbc_unified.py','--config',f'cfg/xwbc/v02/unified_{v}.yaml','--output',str(out/v)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 save(out/'training.launch.json',{v:p.pid for v,p in workers.items()})
 for v,p in workers.items():
  if p.wait()!=0:raise RuntimeError(v+' training failed; other variant preserved')
 identities=[]
 for v in variants:
  run=out/v;result=json.loads((run/'result.json').read_text());assert result['updates']==4300
  assert result['optimizer_steps']==132800 and result['replay_samples']==534118400
  subprocess.run([sys.executable,'-B','scripts/audit_xwbc_unified.py','--run',str(run)],check=True)
  identities.append(json.loads((run/'parents.json').read_text())['initial_model_sha256'])
 assert len(set(identities))==1
 save(out/'initialization_audit.json',dict(passed=True,identical_random_initialization=identities[0]))
 for v in variants:
  run=out/v;ck=run/'checkpoint_4300.pt'
  save(run/'selection.json',dict(endpoint=4300,criterion='prespecified_fixed_endpoint',sha256=sha(ck)))
  for split,nenv in [('val',64),('test',500)]:
   dest=run/f'eval_{split}_4300'
   with (run/f'eval_{split}.console.log').open('x') as log:
    subprocess.run([sys.executable,'-u','-B','scripts/xwbc_staged.py','evaluate','--checkpoint',str(ck),'--output',str(dest),'--split',split,'--num-envs',str(nenv),'--seed','42'],stdout=log,stderr=subprocess.STDOUT,check=True)
   baseline=references[split];assert sha(baseline/'summary.json')==protocol['references'][split]['summary_sha256']
   for r in ('g1','roban'):assert json.loads((dest/f'{r}_environment.json').read_text())==json.loads((baseline/f'{r}_environment.json').read_text())
   save(dest/'independent_score_audit.json',dict(candidate=audit_scores(dest,ck),reference=audit_scores(baseline,refck)))
   save(run/f'comparison_{split}.json',compare(baseline,dest))
 for split in ('val','test'):save(out/f'curriculum_vs_uniform_{split}.json',compare(out/'uniform'/f'eval_{split}_4300',out/'curriculum'/f'eval_{split}_4300'))
 save(out/'pipeline_result.json',dict(status='complete',stage='U',endpoint=4300,test_used_for_selection=False))
if __name__=='__main__':main()
