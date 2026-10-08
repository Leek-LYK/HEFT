"""Short E2300 eight-step refinement; engineering only, no fabricated admission."""
import argparse,copy,json,math,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.learning.xwbc_staged import config_for,runtime_setup,seed_all,make_bank,load_model,control_contract,checkpoint
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_incremental import load_checkpoint_safe,eight_step_origin,equal_tree
from active_adaptation.learning.modules.xwbc_staged import Reservoir,frozen_digest,student_mean
from active_adaptation.learning.xwbc_finetune import BalancedSourcePool
from active_adaptation.learning.xwbc import history_options
from active_adaptation.utils.xwbc_data import ROOT,ROBOTS,SOURCES,sha256
from scripts.refine_xwbc_student import configure,collect,update


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 runtime_setup();seed_all(0);out=args.output;out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'logs/v0.2/action_decoder_20261007/formal/flow/E_FLOW/checkpoint_2300.pt'
 old=load_checkpoint_safe(source)
 converted=eight_step_origin(old,source,sha256(source))
 m=XWBCPolicy(**converted['constructor']).to('cuda:0');m.load_state_dict(converted['model'])
 anchor=copy.deepcopy(m).requires_grad_(False);params=configure(m,True);fixed=frozen_digest(m)
 cfg=config_for(ROOT/'cfg/xwbc/v02/refinement_flow_incremental.yaml','E',SimpleNamespace(num_envs=96,updates=3))
 cfg.update(supervised_steps=4,batch_size=128,rollout_steps=4)
 bank=make_bank(cfg,'train');opt=torch.optim.Adam(params,lr=cfg['learning_rate'])
 teachers={r:load_model(ROOT/f'logs/v0.2/staged_run01/teacher_{r}/checkpoint_final.pt')[0].requires_grad_(False) for r in ROBOTS}
 replay={r:Reservoir(4096,cfg['seed']+i) for i,r in enumerate(ROBOTS)}
 assert not opt.state and all(b.size==0 for b in replay.values()) and not m.heads.rng_state()
 pools={};metrics=[]
 try:
  for r in ROBOTS:pools[r]=BalancedSourcePool(r,bank,96,cfg['history'],**history_options(cfg))
  for _ in range(3):
   row={}
   for r,pool in pools.items():row.update({r+'/'+k:v for k,v in collect(m,teachers[r],pool,bank,replay[r],4).items()})
   row.update(update(m,anchor,opt,bank,replay,cfg));metrics.append(row)
  assert all(math.isfinite(v) for row in metrics for v in row.values())
  assert frozen_digest(m)==fixed and equal_tree({k:v.cpu() for k,v in anchor.state_dict().items()},converted['model'])
  parents=dict(baselines=json.loads((ROOT/'logs/v0.2/action_decoder_20261007/formal/flow/E_FLOW/eval_val_steps_8/summary.json').read_text())['matrix'])
  checkpoint(out/'E_smoke.pt',m,opt,'E',3,cfg,old['dataset_digest'],parents,{r:control_contract(p) for r,p in pools.items()},replay)
  loaded,state=load_model(out/'E_smoke.pt');assert loaded.heads.steps==8
  for r,pool in pools.items():
   obs=pool.observe();cmd=bank.gather(obs['motion'],obs['frame'])
   for s in SOURCES:
    a=student_mean(m,r,obs['proprio'],cmd[s],s);b=student_mean(loaded,r,obs['proprio'],cmd[s],s)
    assert torch.isfinite(b).all();torch.testing.assert_close(a,b,rtol=0,atol=0)
  result=dict(passed=True,purpose='engineering_only',updates=3,flow_steps=8,anchor='starting_E2300_steps8_unchanged',optimizer_and_replay='fresh',metrics=metrics)
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))
 finally:
  for pool in pools.values():pool.close()
if __name__=='__main__':main()
