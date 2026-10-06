"""Audit transformer lineage, budgets and checkpoint reload without simulator writes."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.learning.xwbc_staged import load_model,semantic_stage,context_contract
from active_adaptation.utils.xwbc_data import sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
 torch.set_num_threads(1);cfg=json.loads((run/'config.json').read_text());stage=json.loads((run/'result.json').read_text())['stage']
 path=run/'checkpoint_final.pt';model,state=load_model(path,'cpu');assert stage==state['stage'] and stage in ('A_T','B_T','E_T')
 assert state['constructor']['transformer']==cfg['transformer'];assert state['config']==cfg
 assert state['context']==context_contract(cfg) and state['dataset_digest']==sha256(Path(cfg['dataset'])/'dataset.json')
 assert all(torch.isfinite(v).all() for v in model.state_dict().values())
 rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
 assert [r['update'] for r in rows]==list(range(1,cfg['updates']+1)) and state['update']==cfg['updates']
 assert all(math.isfinite(v) for r in rows for v in r.values())
 targets=1 if stage=='A_T' else 2;assert rows[-1]['transitions']==cfg['updates']*cfg['num_envs']*cfg['rollout_steps']*targets
 if stage=='A_T':
  expected=cfg['updates']*cfg['epochs']*math.ceil(cfg['num_envs']*cfg['rollout_steps']/cfg['minibatch_size'])
 elif stage=='B_T':expected=cfg['updates']*cfg['supervised_steps']
 else:expected=rows[-1]['optimizer_steps']
 actual=max(int(v['step']) for v in state['optimizer']['state'].values());assert actual==expected,(actual,expected)
 for name,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha256(run/'code'/name)==h
 if stage=='B_T':
  for r in ('g1','roban'):
   parent=state['parents'][r];assert sha256(parent['checkpoint'])==parent['sha256']
   teacher=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False);assert teacher['stage']=='A_T'
 if stage=='E_T':
  parent=torch.load(state['parents']['checkpoint'],map_location='cpu',weights_only=False);assert parent['stage']=='B_T'
  assert sha256(state['parents']['checkpoint'])==state['parents']['sha256']
  for k,v in state['model'].items():
   if not k.startswith(('encoders.','proprio.','backbones.','heads.')):assert torch.equal(v,parent['model'][k])
 result=dict(passed=True,stage=stage,updates=state['update'],transitions=rows[-1]['transitions'],optimizer_steps=actual,
             checkpoint_sha256=sha256(path),parameters=sum(p.numel() for p in model.parameters()),transformer=cfg['transformer'])
 with (run/'transformer_audit.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
