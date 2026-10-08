"""XWBC v0.2: native privileged teachers -> Human DAgger -> frozen-anchor encoders."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from active_adaptation.envs.xwbc_staged import StagedPool
from active_adaptation.learning.modules.xwbc import XWBCPolicy,gae,ppo_loss
from active_adaptation.learning.modules.xwbc_staged import (
    NativeTeacher,Reservoir,student_mean,student_features,configure_student,action_mse,stage_c_loss,frozen_digest)
from active_adaptation.learning.xwbc import runtime_setup,seed_all,write_json,aggregate,history_options,load_train_config
from active_adaptation.utils.xwbc_data import CommandBank,ROOT,ROBOTS,SOURCES,sha256

PROTOCOL='xwbc-v02-staged-1'

def semantic_stage(stage):
    return stage[:-2] if stage.endswith('_T') else stage

def stage_name(stage,cfg):
    return stage+'_T' if cfg.get('transformer') is not None and stage in ('A','B','E') else stage



def config_for(path,stage,args):
    cfg=load_train_config(path)
    if stage=='A':cfg.update(cfg['stage_a'])
    for key in ['dataset','num_envs','updates','seed']:
        value=getattr(args,key,None)
        if value is not None:cfg[key]=value
    cfg['dataset']=str(Path(cfg['dataset']).resolve())
    for key in ('num_envs','updates','rollout_steps','checkpoint_interval'):
        if cfg[key]<=0:raise ValueError(key+' must be positive')
    return cfg


def make_bank(cfg,split,robot=None):
    return CommandBank(cfg['dataset'],split,'cuda:0',offsets=cfg['command_offsets'],sources=None if robot is None else [robot])


def context_contract(cfg):
    return {k:cfg[k] for k in ['command_offsets','history_steps','action_history']}


def control_contract(pool):
    a=pool.env.action_manager
    return dict(joint_names=list(a.joint_names),scale=a.action_scaling.tolist(),
                default_position=a.default_joint_pos[0,a.joint_ids].tolist(),
                action=pool.resolved_config['action'],asset=pool.robot_name)


def checkpoint(path,model,optimizer,stage,update,cfg,digest,parents,controls,replay=None):
    state=dict(format=PROTOCOL,stage=stage_name(stage,cfg),model=model.state_dict(),constructor=model.constructor,
               optimizer=optimizer.state_dict(),update=update,config=cfg,dataset_digest=digest,
               parents=parents,controls=controls,context=context_contract(cfg),
               torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
               numpy_rng=np.random.get_state(),python_rng=random.getstate(),
               replay=None if replay is None else {r:b.state_dict() for r,b in replay.items()},
               restart_semantics='restore learner/optimizer/reservoir/RNG; restart live simulator episodes')
    if hasattr(getattr(model,'heads',None),'rng_state'):
        state['decoder_rng']=model.heads.rng_state()
    torch.save(state,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)


def load_model(path,device='cuda:0'):
    from active_adaptation.learning.modules.xwbc_incremental import load_checkpoint_safe
    state=load_checkpoint_safe(path)
    if state.get('format')!=PROTOCOL:raise ValueError('Expected a v0.2 staged checkpoint')
    model=(NativeTeacher(**state['constructor']) if semantic_stage(state['stage'])=='A' else XWBCPolicy(**state['constructor'])).to(device)
    model.load_state_dict(state['model']);model.eval()
    return model,state


def initialize_run(output,cfg):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg)
    write_json(out/'provenance.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        device=torch.cuda.get_device_name(),argv=os.sys.argv,protocol=PROTOCOL,payload_enabled=False))
    files=['active_adaptation/learning/xwbc_staged.py','active_adaptation/learning/xwbc_finetune.py','active_adaptation/learning/modules/xwbc_staged.py',
           'active_adaptation/envs/xwbc_staged.py','active_adaptation/envs/xwbc.py',
           'active_adaptation/learning/modules/xwbc.py','active_adaptation/learning/xwbc.py',
           'active_adaptation/learning/modules/xwbc_incremental.py',
           'active_adaptation/utils/xwbc_data.py','active_adaptation/utils/helpers.py',
           'active_adaptation/learning/ppo/common.py','scripts/xwbc_staged.py',
           'scripts/data_process/build_xwbc_dataset.py',
           'cfg/xwbc/v02/staged.yaml','cfg/xwbc/v02/source_adaptation.yaml','cfg/xwbc/v02/finetune.yaml','scripts/run_xwbc_staged.sh','scripts/data_process/select_xwbc_simple.py']
    if cfg.get('transformer') is not None:
        files += ['active_adaptation/learning/modules/xwbc_transformer.py','scripts/run_xwbc_transformer.py','scripts/audit_xwbc_transformer.py','scripts/refine_xwbc_student.py','cfg/xwbc/v02/staged_transformer.yaml','cfg/xwbc/v02/refinement_transformer.yaml']
    if cfg.get('student_fsq') is not None:
        files += ['active_adaptation/learning/modules/student_latent_quantizer.py',
                  'scripts/run_xwbc_fsq.py', 'scripts/audit_xwbc_fsq.py',
                  'scripts/refine_xwbc_student.py', 'scripts/compare_xwbc_refinement.py',
                  'scripts/finish_xwbc_no_c_evaluation.py', 'scripts/run_xwbc_no_c_ablation.py',
                  'scripts/smoke_xwbc_fsq.py', 'tests/test_xwbc_fsq.py',
                  'cfg/xwbc/v02/staged_fsq.yaml', 'cfg/xwbc/v02/refinement_fsq.yaml']
    if cfg.get('action_decoder') is not None:
        files += ['active_adaptation/learning/modules/xwbc_action_decoder.py',
                  'scripts/run_xwbc_flow.py','scripts/audit_xwbc_action_decoder.py',
                  'scripts/smoke_xwbc_action_decoder.py','tests/test_xwbc_action_decoder.py',
                  'cfg/xwbc/v02/staged_flow.yaml','cfg/xwbc/v02/refinement_flow.yaml',
                  'cfg/xwbc/v02/staged_decoder_mlp.yaml','cfg/xwbc/v02/refinement_decoder_mlp.yaml']
    if cfg.get('incremental_experiment'):
        files += ['active_adaptation/learning/modules/xwbc_incremental.py',
                  'scripts/run_xwbc_incremental.py','scripts/audit_xwbc_incremental.py','scripts/smoke_xwbc_incremental.py',
                  'tests/test_xwbc_incremental.py',
                  'cfg/xwbc/v02/teacher_resume_15k.yaml','cfg/xwbc/v02/refinement_flow_incremental.yaml']
    for f in files:
        dst=out/'code'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dst)
    write_json(out/'source_hashes.json',{f:sha256(ROOT/f) for f in files})
    (out/'working_tree.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
    from torch.utils.tensorboard import SummaryWriter
    return out,SummaryWriter(str(out/'tensorboard'))


def identity_hash(rows):
    return hashlib.sha256('\n'.join(row['motion_id'] for row in rows).encode()).hexdigest()


def gate_decision(summary):
    """Engineering acceptance on full validation; thresholds are recorded, not universal."""
    issues=[];diagnostics=[];a=summary['acceptance'];stage=semantic_stage(summary['stage'])
    world_issues=diagnostics if a.get('world_error_is_diagnostic',False) else issues
    if summary['split']!='val' or not summary['full_scope']:issues.append('full_validation_required')
    if stage=='A':expected=[summary['robot']+'->'+summary['robot']]
    elif stage=='B':expected=['human->'+r for r in ROBOTS]
    else:expected=[s+'->'+r for r in ROBOTS for s in SOURCES]
    if set(summary['matrix'])!=set(expected):issues.append('route_coverage')
    for route in expected:
        if route not in summary['matrix']:continue
        m=summary['matrix'][route];robot=route.split('->')[1]
        numeric=[m[k] for k in ('clips','success_rate','completion','recovery_clips','mpkpe_root_relative_m','mpkpe_world_m')]
        if not all(np.isfinite(x) for x in numeric):
            issues.append(route+':nonfinite_metric');continue
        if m['clips']!=summary['expected_clips']:issues.append(route+':clip_coverage')
        if m['recovery_clips']/max(1,m['clips'])>a['max_recovery_clip_fraction']:issues.append(route+':recovery')
        if stage=='A':
            if m['success_rate']<a['teacher_success']:issues.append(route+':success')
            if m['completion']<a['teacher_completion']:issues.append(route+':completion')
            if m['mpkpe_root_relative_m']>a['teacher_relative_mpkpe']:issues.append(route+':tracking_error')
            if m['mpkpe_world_m']>a.get('teacher_world_mpkpe',.15):world_issues.append(route+':world_tracking_error')
        else:
            base=summary['baselines'][route] if stage in ('D','E','U') else summary['baselines'][robot]
            # The capability gate applies only to admission from B into C.
            # C still measures each new route against the approved Human anchor.
            absolute=a.get('stage_b_absolute') if stage=='B' else None
            relative_issues=diagnostics if absolute is not None else issues
            if absolute is not None:
                if m['success_rate']<absolute['success']:issues.append(route+':success')
                if m['completion']<absolute['completion']:issues.append(route+':completion')
                if m['mpkpe_root_relative_m']>absolute['relative_mpkpe']:issues.append(route+':tracking_error')
            if m['success_rate']<base['success_rate']-a['student_success_drop']:relative_issues.append(route+':success_drop')
            if m['completion']<base['completion']-a['student_completion_drop']:relative_issues.append(route+':completion_drop')
            if m['mpkpe_root_relative_m']>base['mpkpe_root_relative_m']+a['student_relative_error_increase_m']:relative_issues.append(route+':tracking_error_increase')
            if m['mpkpe_world_m']>base['mpkpe_world_m']+a.get('student_world_error_increase_m',.05):world_issues.append(route+':world_tracking_error_increase')
    return dict(passed=not issues,issues=issues,diagnostics=diagnostics)


def verify_gate(path,checkpoint_path,cfg,expected_stage,expected_robot=None):
    gate_path=Path(path);gate=json.loads(gate_path.read_text())
    summary_path=gate_path.parent/'summary.json'
    if sha256(summary_path)!=gate['summary_sha256']:raise ValueError('Gate summary digest changed')
    summary=json.loads(summary_path.read_text())
    if summary['protocol']!=PROTOCOL or summary['checkpoint_sha256']!=sha256(checkpoint_path):raise ValueError('Gate/checkpoint mismatch')
    if summary['dataset_digest']!=sha256(Path(cfg['dataset'])/'dataset.json'):raise ValueError('Gate dataset mismatch')
    if summary['stage']!=stage_name(expected_stage,cfg) or (expected_robot and summary['robot']!=expected_robot):raise ValueError('Gate stage/robot mismatch')
    if summary['context']!=context_contract(cfg):raise ValueError('Gate context mismatch')
    if summary['acceptance']!=cfg['acceptance']:raise ValueError('Gate acceptance settings changed')
    schema=json.loads((Path(cfg['dataset'])/'dataset.json').read_text())
    rows=[]
    for sh in schema['splits']['val']:rows.extend(json.loads((Path(cfg['dataset'])/sh['path']/'motions.json').read_text()))
    if summary['motion_ids_digest']!=identity_hash(rows):raise ValueError('Validation motion set mismatch')
    decision=gate_decision(summary)
    if not decision['passed']:raise ValueError('Stage prerequisite not reached: '+', '.join(decision['issues']))
    return summary


def validate_parent(state,cfg,stage):
    if state['stage']!=stage_name(stage,cfg):raise ValueError(f'Expected stage {stage}')
    if stage != 'A' and state['constructor'].get('student_fsq') != cfg.get('student_fsq'):
        raise ValueError('Parent student FSQ configuration differs')
    if stage != 'A' and state['constructor'].get('action_decoder') != cfg.get('action_decoder'):
        raise ValueError('Parent action decoder configuration differs')
    if state['context']!=context_contract(cfg):raise ValueError('Parent time context differs')
    if state['dataset_digest']!=sha256(Path(cfg['dataset'])/'dataset.json'):raise ValueError('Parent dataset differs')


def create_pool(robot,bank,cfg,evaluation=False):
    return StagedPool(robot,bank,cfg['num_envs'],cfg['history'],evaluation=evaluation,**history_options(cfg))


def log_metrics(log,writer,row):
    if not all(np.isfinite(v) for v in row.values()):raise FloatingPointError('Nonfinite learner metrics')
    log.write(json.dumps(row)+'\n')
    for k,v in row.items():writer.add_scalar(k,v,row['update'])
    writer.flush();print(json.dumps(row),flush=True)


@torch.no_grad()
def teacher_rollout(model,pool,bank,steps):
    samples=[]
    for _ in range(steps):
        obs=pool.observe();priv=pool.teacher_observation(obs);cmd=bank.gather(obs['motion'],obs['frame'])[model.robot]
        dist=model.distribution(obs['proprio'],priv,cmd);action=dist.sample()
        value=model.value(obs['proprio'],priv,cmd)
        transition,nxt=pool.step(action,dist.mean)
        nxt_priv=pool.teacher_observation(nxt);nxt_cmd=bank.gather(nxt['motion'],nxt['frame'])[model.robot]
        next_value=model.value(nxt['proprio'],nxt_priv,nxt_cmd)
        done=transition['done'].squeeze(-1)
        terminal=transition['terminated'].squeeze(-1)|(pool.command.t>=pool.command.lengths-1)
        row=dict(proprio=obs['proprio'],privileged=priv,motion=obs['motion'],frame=obs['frame'],
                 action=action,logp=dist.log_prob(action).sum(-1),value=value,next_value=next_value,
                 reward=transition['reward'].sum(-1),done=done,terminal=terminal,
                 recovered=pool.env._nonfinite_state_mask.clone())
        samples.append({k:v.clone() for k,v in row.items()});pool.reset(done.nonzero().flatten())
    batch={k:torch.stack([r[k] for r in samples]) for k in samples[0]}
    adv,ret=gae(batch['reward'],batch['value'],batch['next_value'],batch['done'],batch['terminal'])
    batch['advantage']=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6);batch['returns']=ret
    for k,v in batch.items():
        if v.is_floating_point() and not torch.isfinite(v).all():raise FloatingPointError('Teacher rollout '+k)
    return {k:v.flatten(0,1) for k,v in batch.items()}


def teacher_update(model,optimizer,bank,batch,cfg):
    metrics=[];n=len(batch['reward'])
    for _ in range(cfg['epochs']):
        perm=torch.randperm(n,device=bank.device)
        for start in range(0,n,cfg['minibatch_size']):
            b={k:v[perm[start:start+cfg['minibatch_size']]] for k,v in batch.items()}
            cmd=bank.gather(b['motion'],b['frame'])[model.robot]
            dist=model.distribution(b['proprio'],b['privileged'],cmd)
            pg,ent,kl=ppo_loss(dist,b['action'],b['logp'],b['advantage'],cfg['clip'])
            vl=(model.value(b['proprio'],b['privileged'],cmd)-b['returns']).square().mean()
            loss=pg+cfg['value_weight']*vl-cfg['entropy_weight']*ent
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();metrics.append(dict(policy_loss=float(pg.detach()),value_loss=float(vl.detach()),entropy=float(ent.detach()),kl=float(kl.detach()),grad_norm=float(grad)))
    return {k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]}


def restore_optimizer(args,model,optimizer,cfg,stage,digest,replay=None,parents=None,controls=None):
    if not args.resume:return 0
    restored,state=load_model(args.resume,device=next(model.parameters()).device)
    if state['stage']!=stage_name(stage,cfg) or state['dataset_digest']!=digest or restored.constructor!=model.constructor:raise ValueError('Resume model/stage/data mismatch')
    if controls is not None and state['controls']!=controls:raise ValueError('Resume control contract changed')
    if stage=='B' and any(state['parents'][r]['sha256']!=parents[r]['sha256'] for r in ROBOTS):raise ValueError('Resume teachers changed')
    if stage=='C' and state['parents']['sha256']!=parents['sha256']:raise ValueError('Resume Human anchor changed')
    for key in ['num_envs','rollout_steps','command_offsets','history_steps','action_history','learning_rate']:
        if cfg[key]!=state['config'][key]:raise ValueError('Resume configuration changed: '+key)
    active_keys=(['epochs','minibatch_size','clip','entropy_weight','value_weight'] if stage=='A'
                 else ['replay_capacity','supervised_steps','batch_size','stage_b' if stage=='B' else 'stage_c'])
    for key in [*active_keys,'max_grad_norm','seed']:
        if cfg.get(key)!=state['config'].get(key):raise ValueError('Resume learning setting changed: '+key)
    model.load_state_dict(restored.state_dict());optimizer.load_state_dict(state['optimizer']);del restored
    if hasattr(model.heads if hasattr(model,'heads') else None,'restore_rng'):
        if 'decoder_rng' not in state:raise ValueError('Missing decoder RNG for exact learner resume')
        model.heads.restore_rng(state['decoder_rng'])
    torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    if replay is not None:
        for r,b in replay.items():b.load_state_dict(state['replay'][r])
    return state['update']


def train_teacher(args):
    runtime_setup();cfg=config_for(args.config,'A',args);seed_all(cfg['seed'])
    out,writer=initialize_run(args.output,cfg);pool=None
    try:
        bank=make_bank(cfg,'train',args.robot);pool=create_pool(args.robot,bank,cfg)
        pool.fixed_source=SOURCES.index(args.robot);pool.reset(pool.all_ids)
        obs=pool.observe();priv=pool.teacher_observation(obs)
        model=NativeTeacher(args.robot,bank.dims[args.robot],pool.proprio_dim,priv.shape[-1],pool.action_dim,cfg['hidden'],transformer=cfg.get('transformer')).to(bank.device)
        write_json(out/'model.json',model.constructor)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        digest=sha256(Path(cfg['dataset'])/'dataset.json');controls={args.robot:control_contract(pool)}
        write_json(out/'environment.json',pool.resolved_config);write_json(out/'control_contract.json',controls)
        start=restore_optimizer(args,model,optimizer,cfg,'A',digest,controls=controls);last=start;t0=time.monotonic()
        teacher_parents={}
        if args.resume:
            teacher_parents=dict(resume_checkpoint=str(Path(args.resume).resolve()),sha256=sha256(args.resume),start_update=start)
            write_json(out/'resume.json',dict(**teacher_parents,target_update=cfg['updates'],additional_updates=cfg['updates']-start,
                       optimizer_restored=True,rng_restored=True,simulator_episodes='restarted'))
            if cfg.get('incremental_experiment'):
                checkpoint(out/'checkpoint_start.pt',model,optimizer,'A',start,cfg,digest,teacher_parents,controls)
        if start>=cfg['updates']:raise ValueError('Update target must exceed the resumed update')
        with (out/'metrics.jsonl').open('w',buffering=1) as log:
            for i in range(start,cfg['updates']):
                tick=time.monotonic();batch=teacher_rollout(model,pool,bank,cfg['rollout_steps'])
                m=teacher_update(model,optimizer,bank,batch,cfg);last=i+1
                m.update(update=last,transitions=last*cfg['num_envs']*cfg['rollout_steps'],
                         reward=batch['reward'].mean().item(),done_fraction=batch['done'].float().mean().item(),
                         recovery_fraction=batch['recovered'].float().mean().item(),recoveries=pool.env.nonfinite_state_recovery_count,
                         elapsed_seconds=time.monotonic()-t0,transitions_per_second=len(batch['reward'])/(time.monotonic()-tick))
                if args.resume:m.update(incremental_updates=last-start,incremental_transitions=(last-start)*cfg['num_envs']*cfg['rollout_steps'])
                log_metrics(log,writer,m)
                if last%cfg['checkpoint_interval']==0:checkpoint(out/f'checkpoint_{last}.pt',model,optimizer,'A',last,cfg,digest,teacher_parents,controls)
                if args.max_minutes and time.monotonic()-t0>=args.max_minutes*60:break
        checkpoint(out/'checkpoint_final.pt',model,optimizer,'A',last,cfg,digest,teacher_parents,controls)
        write_json(out/'result.json',dict(status='complete',stage=stage_name('A',cfg),robot=args.robot,updates=last,requested_updates=cfg['updates'],
                                        elapsed_seconds=time.monotonic()-t0,ready_for_B='requires full validation gate',start_update=start,additional_updates=last-start,additional_transitions=(last-start)*cfg['num_envs']*cfg['rollout_steps']))
    finally:
        writer.close()
        if pool is not None:pool.close()


@torch.no_grad()
def query_teacher(teacher,pool,obs,commands):
    if teacher.robot!=pool.robot_name:raise ValueError('Teacher queried on wrong target robot')
    return teacher.distribution(obs['proprio'],pool.teacher_observation(obs),commands[pool.robot_name]).mean.detach()


def takeover_probability(stage,iteration,cfg):
    if stage=='B':
        warm=cfg['stage_b']['warmup_updates'];decay=cfg['stage_b']['teacher_decay_updates']
        return 1. if iteration<warm else max(0.,1-(iteration-warm)/max(decay,1))
    return max(0.,1-iteration/max(cfg['stage_c']['human_decay_updates'],1))


@torch.no_grad()
def dagger_collect(stage,model,teachers,pool,bank,replay,steps,beta):
    robot=pool.robot_name;takeovers=0;rewards=[];action_errors=[]
    for _ in range(steps):
        obs=pool.observe();commands=bank.gather(obs['motion'],obs['frame'])
        if stage=='B':
            prediction=student_mean(model,robot,obs['proprio'],commands['human'],'human')
            target=query_teacher(teachers[robot],pool,obs,commands)
        else:
            target=student_mean(model,robot,obs['proprio'],commands['human'],'human').detach()
            g=student_mean(model,robot,obs['proprio'],commands['g1'],'g1')
            r=student_mean(model,robot,obs['proprio'],commands['roban'],'roban')
            prediction=torch.where((obs['source']==1)[:,None],g,r)
        # A single real trajectory; both queries happened before exactly this step.
        replay.add(dict(proprio=obs['proprio'],motion=obs['motion'],frame=obs['frame'],target=target))
        action_errors.append(float(action_mse(prediction,target)))
        use_anchor=torch.rand(pool.num_envs,device=bank.device)<beta
        executed=torch.where(use_anchor[:,None],target,prediction)
        transition,_=pool.step(executed,executed)
        takeovers+=int(use_anchor.sum());rewards.append(float(transition['reward'].sum(-1).mean()))
        ids=transition['done'].squeeze(-1).nonzero().flatten();pool.reset(ids)
        if stage=='C':pool.source[ids]=torch.randint(1,3,(len(ids),),device=bank.device)
    return dict(takeover_fraction=takeovers/(steps*pool.num_envs),mixture_reward=float(np.mean(rewards)),
                visited_state_action_mse=float(np.mean(action_errors)))


def supervised_update(stage,model,optimizer,bank,replays,cfg):
    metrics=[]
    for _ in range(cfg['supervised_steps']):
        optimizer.zero_grad(set_to_none=True);row={}
        for robot,replay in replays.items():
            b=replay.sample(cfg['batch_size'],bank.device);commands=bank.gather(b['motion'],b['frame'])
            if stage=='B':
                if getattr(model.heads,'kind',None)=='flow':
                    hidden=student_features(model,robot,b['proprio'],commands['human'],'human')
                    loss=model.heads.matching_loss(robot,hidden,b['target'])
                    row[robot+'/flow_mse']=float(loss.detach())
                    with torch.no_grad():mean=model.heads.predict(robot,hidden.detach())
                else:
                    mean=student_mean(model,robot,b['proprio'],commands['human'],'human')
                    loss=action_mse(mean,b['target'])
                row[robot+'/action_mse']=float(action_mse(mean,b['target']).detach())
            else:
                loss,z,act=stage_c_loss(model,robot,b['proprio'],commands,b['target'],cfg['stage_c']['token_weight'],cfg['stage_c']['action_weight'])
                row[robot+'/token_mse']=float(z.detach());row[robot+'/action_mse']=float(act.detach())
            (loss/len(replays)).backward() # equal target weights AFTER valid-dimension mean
        grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step();row['gradient_norm']=float(grad);metrics.append(row)
    return {k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]}


def train_student(args,stage):
    runtime_setup();cfg=config_for(args.config,stage,args);seed_all(cfg['seed'])
    teachers={};parents={}
    if stage=='B':
        for robot in ROBOTS:
            path=getattr(args,robot+'_teacher');gate=getattr(args,robot+'_gate')
            summary=verify_gate(gate,path,cfg,'A',robot)
            teachers[robot],state=load_model(path);validate_parent(state,cfg,'A')
            teachers[robot].requires_grad_(False)
            parents[robot]=dict(checkpoint=str(Path(path).resolve()),sha256=sha256(path),gate=str(Path(gate).resolve()),
                                baseline=summary['matrix'][robot+'->'+robot],controls=state['controls'][robot])
    else:
        summary=verify_gate(args.student_gate,args.student,cfg,'B')
        model,state=load_model(args.student);validate_parent(state,cfg,'B')
        parents=dict(student=str(Path(args.student).resolve()),sha256=sha256(args.student),gate=str(Path(args.student_gate).resolve()),
                     baselines={r:summary['matrix']['human->'+r] for r in ROBOTS},controls=state['controls'])
    out,writer=initialize_run(args.output,cfg);pools={}
    try:
        bank=make_bank(cfg,'train');digest=sha256(Path(cfg['dataset'])/'dataset.json')
        for robot in ROBOTS:
            pool=create_pool(robot,bank,cfg);pools[robot]=pool
            pool.fixed_source=0 if stage=='B' else None;pool.reset(pool.all_ids)
            if stage=='C':pool.source=torch.randint(1,3,(pool.num_envs,),device=bank.device)
            expected=parents[robot]['controls'] if stage=='B' else parents['controls'][robot]
            if control_contract(pool)!=expected:raise ValueError('Teacher/anchor action contract changed: '+robot)
            write_json(out/f'{robot}_environment.json',pool.resolved_config)
        if stage=='B':
            model=XWBCPolicy(bank.dims,{r:p.proprio_dim for r,p in pools.items()},{r:p.action_dim for r,p in pools.items()},
                              cfg['token_dim'],cfg['hidden'],'shared',encoder_hidden=cfg['encoder_hidden'],transformer=cfg.get('transformer'),student_fsq=cfg.get('student_fsq'),action_decoder=cfg.get('action_decoder')).to(bank.device)
        params=configure_student(model,stage);optimizer=torch.optim.Adam(params,lr=cfg['learning_rate'])
        replay={r:Reservoir(cfg['replay_capacity'],cfg['seed']+i) for i,r in enumerate(ROBOTS)}
        anchor_frozen=frozen_digest(model) if stage=='C' else None
        start=restore_optimizer(args,model,optimizer,cfg,stage,digest,replay,parents)
        if start>=cfg['updates']:raise ValueError('Update target must exceed the resumed update')
        reference_frozen=frozen_digest(model)
        if stage=='B' and cfg.get('student_fsq') is not None:
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'initial_model.pt')
        if stage=='C' and reference_frozen!=anchor_frozen:raise ValueError('Resume executor differs from approved Human anchor')
        write_json(out/'model.json',model.constructor)
        write_json(out/'trainable_parameters.json',[name for name,p in model.named_parameters() if p.requires_grad])
        write_json(out/'frozen_modules.json',dict(sha256=reference_frozen,stage=stage))
        controls={r:control_contract(p) for r,p in pools.items()};t0=time.monotonic();last=start
        with (out/'metrics.jsonl').open('w',buffering=1) as log:
            for i in range(start,cfg['updates']):
                tick=time.monotonic();beta=takeover_probability(stage,i,cfg);m={}
                for r,pool in pools.items():
                    stats=dagger_collect(stage,model,teachers,pool,bank,replay[r],cfg['rollout_steps'],beta)
                    m.update({r+'/'+k:v for k,v in stats.items()});m[r+'/replay_seen']=replay[r].seen
                    m[r+'/recoveries']=pool.env.nonfinite_state_recovery_count
                m.update(supervised_update(stage,model,optimizer,bank,replay,cfg));last=i+1
                m.update(update=last,transitions=last*cfg['num_envs']*cfg['rollout_steps']*2,beta=beta,
                         elapsed_seconds=time.monotonic()-t0,iteration_seconds=time.monotonic()-tick)
                log_metrics(log,writer,m)
                if last%cfg['checkpoint_interval']==0:
                    if frozen_digest(model)!=reference_frozen:raise RuntimeError('Frozen executor changed')
                    checkpoint(out/f'checkpoint_{last}.pt',model,optimizer,stage,last,cfg,digest,parents,controls,replay)
                if args.max_minutes and time.monotonic()-t0>=args.max_minutes*60:break
        if frozen_digest(model)!=reference_frozen:raise RuntimeError('Frozen executor changed')
        checkpoint(out/'checkpoint_final.pt',model,optimizer,stage,last,cfg,digest,parents,controls,replay)
        write_json(out/'result.json',dict(status='complete',stage=stage_name(stage,cfg),updates=last,requested_updates=cfg['updates'],
                     frozen_modules='unchanged',ppo_updates=0,elapsed_seconds=time.monotonic()-t0))
    finally:
        writer.close()
        for pool in pools.values():pool.close()


def evaluate(args):
    runtime_setup();seed_all(args.seed)
    model,state=load_model(args.checkpoint);stage=semantic_stage(state['stage']);cfg=copy.deepcopy(state['config'])
    flow_steps=getattr(args,'flow_steps',None)
    if flow_steps is not None:
        if args.split!='val' or getattr(getattr(model,'heads',None),'kind',None)!='flow' or flow_steps<=0:
            raise ValueError('Flow step override requires a flow model and validation split')
        model.heads.steps=flow_steps
    if args.dataset:cfg['dataset']=str(Path(args.dataset).resolve())
    if state['dataset_digest']!=sha256(Path(cfg['dataset'])/'dataset.json'):raise ValueError('Evaluation dataset differs')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    teacher_robot=model.robot if stage=='A' else None
    bank=make_bank(cfg,args.split,teacher_robot);count=min(len(bank.rows),args.max_motions or len(bank.rows))
    robots=[teacher_robot] if stage=='A' else list(ROBOTS);results=[]
    action_diagnostics={} if cfg.get('action_decoder') else None
    for robot in robots:
        local_cfg=dict(cfg,num_envs=min(args.num_envs,count));pool=create_pool(robot,bank,local_cfg,evaluation=True)
        try:
            if control_contract(pool)!=state['controls'][robot]:raise ValueError('Evaluation action contract differs')
            write_json(out/f'{robot}_environment.json',pool.resolved_config)
            sources=[robot] if stage=='A' else (['human'] if stage=='B' else list(SOURCES))
            for source in sources:
                pool.fixed_source=SOURCES.index(source)
                if action_diagnostics is not None:
                    diag=action_diagnostics[source+'->'+robot]=dict(requested_delta_sq=0.,applied_pd_target_delta_sq_rad=0.,transitions=0)
                for start in range(0,count,pool.num_envs):
                    real=min(pool.num_envs,count-start);motion=torch.arange(start,start+pool.num_envs,device=bank.device).clamp_max(count-1)
                    pool.command.fixed_motion_ids=motion;seed_all(args.seed+start);pool.reset(pool.all_ids)
                    active=torch.arange(pool.num_envs,device=bank.device)<real
                    sums={k:torch.zeros(pool.num_envs,device=bank.device) for k in pool.metrics()};steps=torch.zeros(pool.num_envs,device=bank.device,dtype=torch.long)
                    previous_requested=None
                    while active.any():
                        with torch.no_grad():
                            obs=pool.observe();commands=bank.gather(obs['motion'],obs['frame'])
                            mean=query_teacher(model,pool,obs,commands) if stage=='A' else student_mean(model,robot,obs['proprio'],commands[source],source)
                            if action_diagnostics is not None:
                                previous_applied=pool.env.action_manager.applied_action.clone()
                            transition,_=pool.step(mean,mean)
                            if action_diagnostics is not None and previous_requested is not None:
                                diag['requested_delta_sq']+=float((mean-previous_requested).square().mean(-1)[active].sum())
                                applied_delta=(pool.env.action_manager.applied_action-previous_applied)*pool.env.action_manager.action_scaling
                                diag['applied_pd_target_delta_sq_rad']+=float(applied_delta.square().mean(-1)[active].sum())
                                diag['transitions']+=int(active.sum())
                            previous_requested=mean.clone()
                            for k,v in pool.metrics().items():sums[k][active]+=v[active]
                            steps[active]+=1;complete=pool.command.t>=pool.command.lengths-1;recovered=pool.env._nonfinite_state_mask
                            finished=active&(transition['done'].squeeze(-1)|complete)
                            for i in finished.nonzero().flatten().tolist():
                                success=bool(complete[i] and not recovered[i]);row=bank.rows[int(motion[i])]
                                reason=[k for k in pool.env.termination_funcs if float(transition['stats','termination',k][i])>0]
                                result=dict(motion_id=row['motion_id'],family_id=row['family_id'],category=row['category'],robot=robot,source=source,
                                            success=success,steps=int(steps[i]),completion=min(1.,int(steps[i])/max(1,int(pool.command.lengths[i])-1)),
                                            reason='complete' if success else ('nonfinite' if recovered[i] else ','.join(reason)),
                                            errors={k:float(v[i]/steps[i]) for k,v in sums.items()})
                                results.append(result)
                                with (out/'clips.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                            active[finished]=False
                print(f'{stage}: evaluated {source}->{robot} {count} clips',flush=True)
        finally:pool.close()
    matrix={f'{s}->{r}':aggregate([x for x in results if x['robot']==r and x['source']==s]) for r in robots
            for s in ([r] if stage=='A' else (['human'] if stage=='B' else SOURCES))}
    for route,metrics in matrix.items():
        source,robot=route.split('->')
        subset=[x for x in results if x['robot']==robot and x['source']==source]
        metrics['categories']={c:aggregate([x for x in subset if x['category']==c]) for c in sorted({x['category'] for x in subset})}
        metrics['macro_category_success']=float(np.mean([v['success_rate'] for v in metrics['categories'].values()]))
    summary=dict(protocol=PROTOCOL,stage=state['stage'],robot=teacher_robot,checkpoint=str(Path(args.checkpoint).resolve()),
                 checkpoint_sha256=sha256(args.checkpoint),dataset_digest=state['dataset_digest'],context=state['context'],
                 split=args.split,full_scope=count==len(bank.rows),expected_clips=len(bank.rows),
                 motion_ids_digest=identity_hash(bank.rows[:count]),matrix=matrix,acceptance=cfg['acceptance'],
                 baselines={} if stage=='A' else ({r:state['parents'][r]['baseline'] for r in ROBOTS} if stage=='B' else state['parents']['baselines']))
    summary['baselines_split']='val' if stage!='A' else None
    if stage in ('D','E','U') and args.split=='val':
        comparison_key = 'comparison_to_reference' if state['parents'].get('evaluation_reference') else 'comparison_to_C'
        summary['evaluation_reference'] = state['parents'].get('evaluation_reference')
        summary[comparison_key]={route:{
            'success_delta_pp':100*(m['success_rate']-summary['baselines'][route]['success_rate']),
            'completion_delta_pp':100*(m['completion']-summary['baselines'][route]['completion']),
            'root_relative_error_delta_m':m['mpkpe_root_relative_m']-summary['baselines'][route]['mpkpe_root_relative_m']
        } for route,m in matrix.items()}
    if cfg.get('action_decoder'):
        summary['action_decoder']=cfg['action_decoder']
        summary['inference']=dict(kind=model.heads.kind,steps=model.heads.steps if model.heads.kind=='flow' else 1,initialization='zero' if model.heads.kind=='flow' else None,primary=flow_steps is None)
    if action_diagnostics is not None:
        for diag in action_diagnostics.values():
            n=max(1,diag['transitions'])
            diag['requested_delta_rms']=(diag['requested_delta_sq']/n)**.5
            diag['applied_pd_target_delta_rms_rad']=(diag['applied_pd_target_delta_sq_rad']/n)**.5
        write_json(out/'action_diagnostics.json',dict(first_step_excluded=True,active_steps_only=True,routes=action_diagnostics))
    write_json(out/'summary.json',summary)
    gate={**gate_decision(summary),'summary_sha256':sha256(out/'summary.json')}
    write_json(out/'gate.json',gate);print(json.dumps(gate),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    for mode in ['teacher','student','sources','finetune']:
        q=sub.add_parser(mode);q.add_argument('--config',default=str(ROOT/('cfg/xwbc/v02/finetune.yaml' if mode=='finetune' else 'cfg/xwbc/v02/staged.yaml')))
        q.add_argument('--output',required=True);q.add_argument('--dataset');q.add_argument('--num-envs',type=int)
        q.add_argument('--updates',type=int);q.add_argument('--seed',type=int);q.add_argument('--resume');q.add_argument('--max-minutes',type=float)
        if mode=='teacher':q.add_argument('--robot',choices=ROBOTS,required=True)
        if mode=='student':
            for r in ROBOTS:
                q.add_argument('--'+r+'-teacher',required=True);q.add_argument('--'+r+'-gate',required=True)
        if mode=='sources':q.add_argument('--student',required=True);q.add_argument('--student-gate',required=True)
        if mode=='finetune':q.add_argument('--policy',required=True);q.add_argument('--policy-gate',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--checkpoint',required=True);q.add_argument('--output',required=True)
    q.add_argument('--dataset');q.add_argument('--split',choices=['val','test'],default='val');q.add_argument('--seed',type=int,default=42)
    q.add_argument('--num-envs',type=int,default=64);q.add_argument('--max-motions',type=int)
    q.add_argument('--flow-steps',type=int,help='Validation-only secondary solver diagnostic')
    args=p.parse_args()
    if args.mode=='teacher':train_teacher(args)
    elif args.mode=='student':train_student(args,'B')
    elif args.mode=='sources':train_student(args,'C')
    elif args.mode=='finetune':
        from active_adaptation.learning.xwbc_finetune import train_finetune
        train_finetune(args)
    else:evaluate(args)
