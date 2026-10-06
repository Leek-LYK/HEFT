"""Audit U lineage, actor-only updates, schedules and exact training budget."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train_xwbc_unified import model_digest,source_probabilities,takeover,learning_rate
from scripts.refine_xwbc_student import iteration_config
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.utils.xwbc_data import sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
 torch.set_num_threads(1);cfg=json.loads((run/'config.json').read_text());n=cfg['updates']
 initial=torch.load(run/'initial_student.pt',map_location='cpu',weights_only=False)
 state=torch.load(run/f'checkpoint_{n}.pt',map_location='cpu',weights_only=False)
 assert state['stage']=='U' and state['update']==n and state['config']==cfg
 assert state['parents']['initialization']=='random' and 'anchor' not in state['parents']
 model=XWBCPolicy(**initial['constructor']);model.load_state_dict(initial['model'])
 assert model_digest(model)==initial['sha256']==state['parents']['initial_model_sha256']
 assert state['constructor']==initial['constructor'] and state['dataset_digest']==sha256(Path(cfg['dataset'])/'dataset.json')
 changes=[]
 for k,v in state['model'].items():
  assert torch.isfinite(v).all()
  if not torch.equal(v,initial['model'][k]):
   assert k.startswith(('encoders.','proprio.','backbones.','heads.'));changes.append(k)
 for s in ('human','g1','roban'):assert any(k.startswith('encoders.shared.'+s+'.') for k in changes)
 for r,record in state['parents']['teachers'].items():
  assert sha256(record['checkpoint'])==record['sha256']
  teacher=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
  assert teacher['stage']=='A' and teacher['controls'][r]==state['controls'][r]
 for name,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha256(run/'code'/name)==h
 rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
 assert [r['update'] for r in rows]==list(range(1,n+1));steps=samples=0
 for row in rows:
  assert all(math.isfinite(v) for v in row.values())
  i=row['update'];c=iteration_config(cfg,i);steps+=c['supervised_steps'];samples+=c['supervised_steps']*c['batch_size']*2
  assert row['optimizer_steps']==steps and row['replay_samples']==samples
  assert row['transitions']==i*cfg['num_envs']*cfg['rollout_steps']*2
  assert row['learning_rate']==learning_rate(cfg,i) and row['takeover_probability']==takeover(cfg,i)
  for j,s in enumerate(('human','g1','roban')):assert row['target_source_'+s]==source_probabilities(cfg,i)[j]
 result=dict(passed=True,stage='U',updates=n,transitions=rows[-1]['transitions'],optimizer_steps=steps,replay_samples=samples,
             initial_model_sha256=initial['sha256'],checkpoint_sha256=sha256(run/f'checkpoint_{n}.pt'),actor_only_changes=True)
 with (run/'independent_audit.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
