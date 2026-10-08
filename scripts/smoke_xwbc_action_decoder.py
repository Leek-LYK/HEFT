"""Engineering smoke: both decoders, teacher takeover, autonomous B/E, and reload."""
import argparse,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.learning.xwbc_staged import (config_for,runtime_setup,seed_all,make_bank,load_model,create_pool,dagger_collect,
 supervised_update,control_contract,checkpoint)
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import configure_student,Reservoir,frozen_digest,student_mean
from active_adaptation.learning.xwbc_finetune import BalancedSourcePool
from active_adaptation.learning.xwbc import history_options
from active_adaptation.utils.xwbc_data import ROOT,ROBOTS,SOURCES,sha256
from scripts.refine_xwbc_student import configure,collect,update


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['flow','decoder_mlp'],required=True);a=p.parse_args();runtime_setup();seed_all(0)
 out=a.root/(a.variant+'_smoke');out.mkdir(parents=True,exist_ok=False)
 cfg=config_for(ROOT/f'cfg/xwbc/v02/staged_{a.variant}.yaml','B',SimpleNamespace(num_envs=96,updates=3))
 cfg.update(supervised_steps=4,batch_size=128,rollout_steps=4)
 bank=make_bank(cfg,'train');teachers={r:load_model(ROOT/f'logs/v0.2/staged_run01/teacher_{r}/checkpoint_final.pt')[0].requires_grad_(False) for r in ROBOTS}
 pools={r:create_pool(r,bank,cfg) for r in ROBOTS};replay={r:Reservoir(4096) for r in ROBOTS}
 for pool in pools.values():pool.fixed_source=0;pool.reset(pool.all_ids)
 cuda_before=torch.cuda.get_rng_state().clone()
 m=XWBCPolicy(bank.dims,{r:p.proprio_dim for r,p in pools.items()},{r:p.action_dim for r,p in pools.items()},cfg['token_dim'],cfg['hidden'],'shared',encoder_hidden=cfg['encoder_hidden'],student_fsq=cfg['student_fsq'],action_decoder=cfg['action_decoder']).to(bank.device)
 assert torch.equal(cuda_before,torch.cuda.get_rng_state()),'Decoder initialization perturbed CUDA RNG'
 params=configure_student(m,'B');before=frozen_digest(m);opt=torch.optim.Adam(params,lr=cfg['learning_rate']);metrics=[]
 controls={r:control_contract(pool) for r,pool in pools.items()}
 try:
  for i in range(3):
   for r,pool in pools.items():dagger_collect('B',m,teachers,pool,bank,replay[r],4,1. if i==0 else 0.)
   metrics.append(supervised_update('B',m,opt,bank,replay,cfg))
  assert frozen_digest(m)==before
  checkpoint(out/'B.pt',m,opt,'B',3,cfg,sha256(Path(cfg['dataset'])/'dataset.json'),{},controls,replay)
 finally:
  for pool in pools.values():pool.close()
 loaded,state=load_model(out/'B.pt');assert state['stage']=='B'
 for k,v in m.state_dict().items():assert torch.equal(v,loaded.state_dict()[k])
 del loaded
 anchor=copy.deepcopy(m).requires_grad_(False);params=configure(m,True);before=frozen_digest(m);opt=torch.optim.Adam(params,lr=1e-5)
 cfg.update(anchor_weight=.1,train_sources=True,learning_rate=1e-5)
 pools={r:BalancedSourcePool(r,bank,96,cfg['history'],**history_options(cfg)) for r in ROBOTS};replay={r:Reservoir(4096) for r in ROBOTS}
 try:
  for i in range(3):
   for r,pool in pools.items():collect(m,teachers[r],pool,bank,replay[r],4)
   metrics.append(update(m,anchor,opt,bank,replay,cfg))
  assert frozen_digest(m)==before
  summary=json.loads((ROOT/'logs/v0.2/no_c_ablation_20261005/direct_e_run01/eval_val_2300/summary.json').read_text())
  parents=dict(baselines=summary['matrix'],evaluation_reference=dict(role='engineering_only_reference'))
  checkpoint(out/'E.pt',m,opt,'E',3,cfg,state['dataset_digest'],parents,controls,replay)
  model,final=load_model(out/'E.pt');assert final['stage']=='E'
  for r,pool in pools.items():
   obs=pool.observe();commands=bank.gather(obs['motion'],obs['frame'])
   for s in SOURCES:assert torch.isfinite(student_mean(model,r,obs['proprio'],commands[s],s)).all()
  assert all(torch.equal(v,anchor.state_dict()[k].cpu()) for k,v in state['model'].items())
  assert all(__import__('math').isfinite(v) for row in metrics for v in row.values())
  result=dict(variant=a.variant,passed=True,purpose='engineering_only_not_quality_admission',B_E_steps=3,actor_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in m.parameters()),frozen_modules='unchanged',all_six_routes_finite=True,metrics=metrics)
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='metrics'}))
 finally:
  for pool in pools.values():pool.close()
if __name__=='__main__':main()
