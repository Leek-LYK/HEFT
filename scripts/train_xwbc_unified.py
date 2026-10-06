"""Stage U: random student, frozen teachers, continuous six-route supervised learning."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.envs.xwbc_staged import StagedPool
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import Reservoir,student_mean,frozen_digest
from active_adaptation.learning.xwbc_staged import (config_for,runtime_setup,seed_all,load_model,validate_parent,
    initialize_run,make_bank,control_contract,query_teacher,checkpoint,log_metrics,write_json,context_contract)
from active_adaptation.learning.xwbc import history_options
from active_adaptation.utils.xwbc_data import ROOT,ROBOTS,SOURCES,sha256
from scripts.refine_xwbc_student import configure,update,iteration_config


def source_probabilities(cfg,iteration):
    if cfg['source_sampling']=='uniform':return [1/3]*3
    if cfg['source_sampling']!='curriculum':raise ValueError('Unknown source schedule')
    t=min(1.,max(0.,(iteration-cfg['curriculum_hold'])/cfg['curriculum_decay']))
    h=.6+(1/3-.6)*t
    return [h,(1-h)/2,(1-h)/2]


def takeover(cfg,iteration):
    return min(1.,max(0.,1-(iteration-1-cfg['teacher_warmup'])/cfg['teacher_decay']))


def learning_rate(cfg,iteration):
    t=min(1.,max(0.,(iteration-1)/(cfg['schedule_updates']-1)))
    return cfg['learning_rate_final']+.5*(cfg['learning_rate']-cfg['learning_rate_final'])*(1+math.cos(math.pi*t))


def assignments(retained,probabilities,total,count):
    """Fill vacancies by largest deficit. Active episodes are never reassigned."""
    counts=list(retained);result=[]
    for _ in range(count):
        s=max(range(3),key=lambda i:total*probabilities[i]-counts[i])
        result.append(s);counts[s]+=1
    return result


class SourcePool(StagedPool):
    def __init__(self,*args,probabilities,**kwargs):
        self.probabilities=probabilities
        super().__init__(*args,**kwargs)
    @torch.no_grad()
    def reset(self,ids):
        if not len(ids):return
        keep=torch.ones(self.num_envs,dtype=torch.bool,device=self.device);keep[ids]=False
        retained=torch.bincount(self.source[keep],minlength=3).cpu().tolist()
        super().reset(ids)
        chosen=assignments(retained,self.probabilities,self.num_envs,len(ids))
        self.source[ids]=torch.tensor(chosen,device=self.device)


def model_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def collect(model,teacher,pool,bank,replay,steps,beta):
    counts=torch.zeros(3,device=bank.device);taken=0;error=reward=0.
    for _ in range(steps):
        obs=pool.observe();commands=bank.gather(obs['motion'],obs['frame'])
        label=query_teacher(teacher,pool,obs,commands)
        means=torch.stack([student_mean(model,pool.robot_name,obs['proprio'],commands[s],s) for s in SOURCES],1)
        action=means[torch.arange(pool.num_envs,device=bank.device),obs['source']]
        replay.add(dict(proprio=obs['proprio'],motion=obs['motion'],frame=obs['frame'],target=label,source=obs['source']))
        mask=torch.rand(pool.num_envs,device=bank.device)<beta
        executed=torch.where(mask[:,None],label,action)
        counts+=torch.bincount(obs['source'],minlength=3);taken+=int(mask.sum())
        error+=float((action-label).square().mean())
        transition,_=pool.step(executed,executed);reward+=float(transition['reward'].sum(-1).mean())
        pool.reset(transition['done'].squeeze(-1).nonzero().flatten())
    result=dict(reward=reward/steps,visited_teacher_mse=error/steps,takeover_fraction=taken/(steps*pool.num_envs))
    replay_counts=torch.bincount(replay.data['source'][:replay.size],minlength=3)
    for i,s in enumerate(SOURCES):
        result['executing_'+s]=float(counts[i]/counts.sum())
        result['replay_'+s]=float(replay_counts[i]/replay.size)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int);p.add_argument('--num-envs',type=int);args=p.parse_args()
    runtime_setup();cfg=config_for(args.config,'U',args);iteration_config(cfg,1)
    assert cfg['anchor_weight']==0 and cfg['train_sources']
    seed_all(cfg['seed']);bank=make_bank(cfg,'train');digest=sha256(Path(cfg['dataset'])/'dataset.json')
    teachers={};teacher_states={};teacher_records={}
    for r in ROBOTS:
        path=ROOT/f'logs/v0.2/staged_run01/teacher_{r}/checkpoint_final.pt'
        teachers[r],teacher_states[r]=load_model(path);validate_parent(teacher_states[r],cfg,'A');assert teachers[r].robot==r
        teachers[r].requires_grad_(False);teacher_records[r]=dict(checkpoint=str(path),sha256=sha256(path))
    refdir=ROOT/cfg['evaluation_reference']/'eval_val_2300';reference=json.loads((refdir/'summary.json').read_text())
    assert reference['dataset_digest']==digest and reference['context']==context_contract(cfg)
    assert reference['split']=='val' and reference['full_scope'] and len(reference['matrix'])==6
    assert json.loads((refdir/'gate.json').read_text())['summary_sha256']==sha256(refdir/'summary.json')
    pools={};writer=None
    try:
        for r in ROBOTS:pools[r]=SourcePool(r,bank,cfg['num_envs'],cfg['history'],probabilities=source_probabilities(cfg,1),**history_options(cfg))
        controls={r:control_contract(pool) for r,pool in pools.items()}
        for r in ROBOTS:assert controls[r]==teacher_states[r]['controls'][r]
        # Reset the initialization RNG immediately before construction, independent of simulation/source sampling.
        seed_all(cfg['init_seed'])
        model=XWBCPolicy(bank.dims,{r:p.proprio_dim for r,p in pools.items()},{r:p.action_dim for r,p in pools.items()},
                         cfg['token_dim'],cfg['hidden'],'shared',encoder_hidden=cfg['encoder_hidden']).to(bank.device)
        initial_digest=model_digest(model);params=configure(model,True);frozen=frozen_digest(model)
        out,writer=initialize_run(args.output,cfg)
        torch.save(dict(model=model.state_dict(),constructor=model.constructor,sha256=initial_digest),out/'initial_student.pt')
        extras=['scripts/train_xwbc_unified.py','scripts/refine_xwbc_student.py','scripts/run_xwbc_unified_experiments.py',
                'scripts/audit_xwbc_unified.py','scripts/compare_xwbc_refinement.py','scripts/finish_xwbc_no_c_evaluation.py',
                'scripts/run_xwbc_no_c_ablation.py',str(Path(args.config).resolve().relative_to(ROOT))]
        hashes=json.loads((out/'source_hashes.json').read_text())
        for f in extras:
            dest=out/'code'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest);hashes[f]=sha256(ROOT/f)
        write_json(out/'source_hashes.json',hashes)
        parents=dict(initialization='random',initial_model_sha256=initial_digest,teachers=teacher_records,baselines=reference['matrix'],
                     evaluation_reference=dict(directory=str(refdir),checkpoint_sha256=reference['checkpoint_sha256'],role='evaluation_only'))
        write_json(out/'parents.json',parents);write_json(out/'trainable_parameters.json',[n for n,p in model.named_parameters() if p.requires_grad])
        for r,pool in pools.items():write_json(out/f'{r}_environment.json',pool.resolved_config)
        optimizer=torch.optim.Adam(params,lr=cfg['learning_rate']);replay={r:Reservoir(cfg['replay_capacity'],cfg['seed']+i) for i,r in enumerate(ROBOTS)}
        seed_all(cfg['seed']);optimizer_steps=samples=0;started=time.monotonic()
        with (out/'metrics.jsonl').open('x',buffering=1) as log:
            for iteration in range(1,cfg['updates']+1):
                row={};probs=source_probabilities(cfg,iteration);beta=takeover(cfg,iteration);lr=learning_rate(cfg,iteration)
                for group in optimizer.param_groups:group['lr']=lr
                for r,pool in pools.items():
                    pool.probabilities=probs
                    row.update({r+'/'+k:v for k,v in collect(model,teachers[r],pool,bank,replay[r],cfg['rollout_steps'],beta).items()})
                    row[r+'/recoveries']=pool.env.nonfinite_state_recovery_count
                active=iteration_config(cfg,iteration);row.update(update(model,None,optimizer,bank,replay,active))
                optimizer_steps+=active['supervised_steps'];samples+=active['supervised_steps']*active['batch_size']*2
                row.update(update=iteration,transitions=iteration*cfg['num_envs']*cfg['rollout_steps']*2,optimizer_steps=optimizer_steps,
                           replay_samples=samples,learning_rate=lr,takeover_probability=beta,supervised_steps=active['supervised_steps'],
                           batch_size=active['batch_size'],elapsed_seconds=time.monotonic()-started)
                row.update({'target_source_'+s:probs[i] for i,s in enumerate(SOURCES)});log_metrics(log,writer,row)
                if iteration%cfg['checkpoint_interval']==0 or iteration==cfg['updates']:
                    assert frozen_digest(model)==frozen
                    checkpoint(out/f'checkpoint_{iteration}.pt',model,optimizer,'U',iteration,cfg,digest,parents,controls,replay)
        write_json(out/'result.json',dict(status='complete',stage='U',updates=iteration,optimizer_steps=optimizer_steps,
                                        replay_samples=samples,ppo_updates=0,frozen_modules='unchanged',elapsed_seconds=time.monotonic()-started))
    finally:
        if writer is not None:writer.close()
        for pool in pools.values():pool.close()


if __name__=='__main__':main()
