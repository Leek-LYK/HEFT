"""Bounded GPU plumbing test with unqualified models; never creates stage approval gates."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from active_adaptation.learning.xwbc_staged import (
    config_for,runtime_setup,seed_all,make_bank,create_pool,dagger_collect,supervised_update)
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import NativeTeacher,Reservoir,configure_student,frozen_digest,student_mean


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='cfg/xwbc/v02/staged.yaml');args=p.parse_args()
    runtime_setup();seed_all(7);cfg=config_for(args.config,'B',args)
    cfg.update(num_envs=8,supervised_steps=2,batch_size=32,rollout_steps=4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);bank=make_bank(cfg,'train');pools={};teachers={}
    try:
        for robot in ['g1','roban']:
            pool=create_pool(robot,bank,cfg);pools[robot]=pool;pool.fixed_source=0;pool.reset(pool.all_ids)
            obs=pool.observe();priv=pool.teacher_observation(obs)
            teachers[robot]=NativeTeacher(robot,bank.dims[robot],pool.proprio_dim,priv.shape[-1],pool.action_dim,hidden=[64,64]).to(bank.device)
            teachers[robot].requires_grad_(False);teachers[robot].eval()
        model=XWBCPolicy(bank.dims,{r:p.proprio_dim for r,p in pools.items()},{r:p.action_dim for r,p in pools.items()},
                        cfg['token_dim'],cfg['hidden'],'shared',encoder_hidden=cfg['encoder_hidden']).to(bank.device)
        params=configure_student(model,'B');opt=torch.optim.Adam(params,lr=1e-4)
        replay={r:Reservoir(128) for r in pools}
        teacher_hash={r:frozen_digest(t) for r,t in teachers.items()}
        for r,pool in pools.items():
            dagger_collect('B',model,teachers,pool,bank,replay[r],4,.5)
            history=pool.history_observation()[:,-pool.action_history*pool.action_dim:].reshape(pool.num_envs,pool.action_history,pool.action_dim)
            torch.testing.assert_close(history[:,0],pool.env.action_manager.applied_action,rtol=0,atol=0)
        bmetrics=supervised_update('B',model,opt,bank,replay,cfg)
        assert all(frozen_digest(t)==teacher_hash[r] for r,t in teachers.items())
        params=configure_student(model,'C');opt=torch.optim.Adam(params,lr=1e-4);before=frozen_digest(model)
        replay={r:Reservoir(128) for r in pools}
        for r,pool in pools.items():
            pool.fixed_source=None;pool.reset(pool.all_ids);pool.source=torch.randint(1,3,(pool.num_envs,),device=bank.device)
            dagger_collect('C',model,{},pool,bank,replay[r],4,0.)
        cmetrics=supervised_update('C',model,opt,bank,replay,cfg)
        assert frozen_digest(model)==before
        report=dict(status='passed',purpose='GPU contract validation only, unqualified synthetic-initialized models',
                    stage_B=bmetrics,stage_C=cmetrics,teachers_frozen=True,executor_frozen=True,
                    actual_action_history='exact match',stage_gates_created=False)
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:
        for p in pools.values():p.close()
if __name__=='__main__':main()
