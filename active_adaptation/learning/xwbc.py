"""Single-stage, two-environment-pool PPO and complete-clip XWBC evaluation."""
from __future__ import annotations

import argparse
import csv
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

from active_adaptation.envs.xwbc import XWBCPool
from active_adaptation.learning.modules.xwbc import XWBCPolicy, alignment_loss, gae, ppo_loss
from active_adaptation.utils.xwbc_data import CommandBank, ROBOTS, SOURCES, ROOT, sha256


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_train_config(path):
    path = Path(path).resolve()
    cfg = OmegaConf.load(path)
    base = cfg.pop('_base_', None)
    if base is not None:
        cfg = OmegaConf.merge(OmegaConf.load(path.parent / base), cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def command_bank(root, split, device, cfg):
    if cfg.get('command_offsets') is None:
        return CommandBank(root, split, device)
    return CommandBank(root, split, device, offsets=cfg['command_offsets'])


def history_options(cfg):
    if cfg.get('history_steps') is None:
        return {}
    return dict(history_steps=cfg['history_steps'], action_history=cfg.get('action_history', 8))


def runtime_setup():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('XWBC simulation requires CUDA')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one process on one GPU; both robots share that process')


def checkpoint(path, model, optimizer, update, cfg, dataset_digest):
    state = dict(format='xwbc-v1', model=model.state_dict(), constructor=model.constructor,
                 optimizer=optimizer.state_dict(), update=update, config=cfg,
                 dataset_digest=dataset_digest, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate(), restart_semantics='optimizer resume with fresh simulator episodes')
    torch.save(state, str(path)+'.tmp')
    os.replace(str(path)+'.tmp', path)


def load_policy(path, device='cuda:0'):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get('format') != 'xwbc-v1':
        raise ValueError('Expected an XWBC checkpoint, not a HEFT stage checkpoint')
    policy = XWBCPolicy(**state['constructor']).to(device)
    policy.load_state_dict(state['model'])
    return policy, state


@torch.no_grad()
def collect(pool, bank, policy, steps):
    samples = []
    reward_sum = 0.
    for _ in range(steps):
        obs = pool.observe()
        commands = bank.gather(obs['motion'], obs['frame'])
        dist, _ = policy.actor(pool.robot_name, obs['proprio'], obs['source'], commands)
        action = dist.sample()
        value = policy.value(pool.robot_name, obs['proprio'], obs['privileged'])
        transition, nxt = pool.step(action, dist.mean)
        next_value = policy.value(pool.robot_name, nxt['proprio'], nxt['privileged'])
        done = transition['done'].squeeze(-1)
        # Motion completion is an episodic terminal; time limits alone bootstrap.
        complete = pool.command.t >= pool.command.lengths-1
        terminal = transition['terminated'].squeeze(-1) | complete
        reward = transition['reward'].sum(-1)
        values = dict(**obs, action=action, old_mean=dist.mean, logp=dist.log_prob(action).sum(-1), value=value,
                      next_value=next_value, reward=reward, done=done, terminal=terminal, completed=complete,
                      terminated=transition['terminated'].squeeze(-1),
                      truncated=transition['truncated'].squeeze(-1),
                      recovered=pool.env._nonfinite_state_mask.clone())
        for i, group in enumerate(pool.env.reward_groups):
            values[f'reward_{group}'] = transition['reward'][:, i]
        for name, tensor in values.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError(f'{pool.robot_name}: nonfinite rollout {name}')
        samples.append({k: v.detach().clone() for k, v in values.items()})
        reward_sum += reward.mean().item()
        pool.reset(done.nonzero().flatten())
    batch = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
    advantage, returns = gae(batch['reward'], batch['value'], batch['next_value'], batch['done'], batch['terminal'])
    batch['advantage'] = (advantage-advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    batch['returns'] = returns
    return {k: v.flatten(0, 1) for k, v in batch.items()}, reward_sum/steps


def update_policy(policy, optimizer, bank, batches, cfg):
    metrics = []
    n = len(next(iter(batches.values()))['value'])
    for _ in range(cfg['epochs']):
        permutations = {r: torch.randperm(n, device=bank.device) for r in batches}
        for start in range(0, n, cfg['minibatch_size']):
            optimizer.zero_grad(set_to_none=True)
            row = {}
            for robot, batch in batches.items():
                idx = permutations[robot][start:start+cfg['minibatch_size']]
                b = {k: v[idx] for k, v in batch.items()}
                commands = bank.gather(b['motion'], b['frame'])
                dist, tokens = policy.actor(robot, b['proprio'], b['source'], commands)
                pg, entropy, kl = ppo_loss(dist, b['action'], b['logp'], b['advantage'], cfg['clip'])
                value = policy.value(robot, b['proprio'], b['privileged'])
                value_loss = (value-b['returns']).square().mean()
                align = alignment_loss(tokens)
                weight = 0. if policy.variant == 'no_align' else cfg['alignment_weight']
                loss = pg + cfg['value_weight']*value_loss - cfg['entropy_weight']*entropy + weight*align
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{robot}: nonfinite PPO loss')
                (loss/len(batches)).backward()
                row.update({f'{robot}/{k}': float(v.detach()) for k, v in
                            dict(policy_loss=pg, value_loss=value_loss, entropy=entropy, kl=kl,
                                 alignment=align, token_std=tokens.std(dim=0, unbiased=False).mean()).items()})
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
            row['gradient_norm'] = float(norm)
            optimizer.step()
            metrics.append(row)
    return {k: float(np.mean([row[k] for row in metrics])) for k in metrics[0]}


def train(args):
    runtime_setup()
    cfg = load_train_config(args.config)
    for name in ('dataset', 'num_envs', 'updates', 'seed', 'variant', 'rollout_steps'):
        value = getattr(args, name, None)
        if value is not None:
            cfg[name] = value
    if min(cfg['num_envs'], cfg['updates'], cfg['rollout_steps'], cfg['epochs'], cfg['minibatch_size']) <= 0:
        raise ValueError('Training budgets must be positive')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    cfg['dataset'] = str(Path(cfg['dataset']).resolve())
    write_json(output/'config.json', cfg)
    seed_all(cfg['seed'])
    bank = command_bank(cfg['dataset'], 'train', 'cuda:0', cfg)
    digest = sha256(Path(cfg['dataset'])/'dataset.json')
    write_json(output/'provenance.json', dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
        dataset_digest=digest, device=torch.cuda.get_device_name(), torch=torch.__version__, argv=os.sys.argv,
        payload_enabled=False, training_stages=['direct_ppo']))
    (output/'working_tree.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'], cwd=ROOT))
    # Preserve new research sources even before their first Git commit.
    source_files = ['active_adaptation/utils/xwbc_data.py', 'active_adaptation/envs/xwbc.py',
                    'active_adaptation/learning/modules/xwbc.py', 'active_adaptation/learning/xwbc.py',
                    'scripts/xwbc.py', 'scripts/run_xwbc.sh', 'scripts/data_process/build_xwbc_dataset.py',
                    'scripts/data_process/select_xwbc_motions.py', 'cfg/xwbc/train.yaml',
                    'cfg/xwbc/heft_context.yaml', 'active_adaptation/utils/helpers.py']
    for filename in source_files:
        destination = output/'code'/filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/filename, destination)
    write_json(output/'source_hashes.json', {p:sha256(ROOT/p) for p in source_files})
    shutil.copyfile(Path(args.config).resolve(), output/'selected_config.yaml')
    write_json(output/'observation_contract.json', dict(command_offsets=bank.offsets.tolist(),
        command_dims=bank.dims, history_steps=cfg.get('history_steps'),
        legacy_history=cfg.get('history', 8) if cfg.get('history_steps') is None else None,
        action_history=cfg.get('action_history', cfg.get('history', 8)), token_dim=cfg['token_dim'],
        encoder_hidden=cfg.get('encoder_hidden', [256]), fps=bank.schema['fps']))
    pools = {}
    writer = None
    try:
        for robot in ROBOTS:
            pools[robot] = XWBCPool(robot, bank, cfg['num_envs'], cfg['history'], **history_options(cfg))
            write_json(output/f'{robot}_environment.json', pools[robot].resolved_config)
        policy = XWBCPolicy(bank.dims, {r:p.proprio_dim for r,p in pools.items()},
                            {r:p.action_dim for r,p in pools.items()}, cfg['token_dim'], cfg['hidden'], cfg['variant'],
                            encoder_hidden=cfg.get('encoder_hidden')).to(bank.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg['learning_rate'])
        start_update = 0
        if args.resume:
            restored, state = load_policy(args.resume)
            if state['dataset_digest'] != digest or restored.constructor != policy.constructor:
                raise ValueError('Resume requires identical dataset and architecture')
            for key in ('num_envs','rollout_steps','learning_rate','epochs','minibatch_size',
                        'clip','entropy_weight','value_weight','alignment_weight','max_grad_norm',
                        'command_offsets','history_steps','action_history','encoder_hidden'):
                if cfg.get(key) != state['config'].get(key):
                    raise ValueError(f'Resume setting changed: {key}; use a separate experiment')
            policy.load_state_dict(restored.state_dict()); del restored
            optimizer.load_state_dict(state['optimizer']); start_update = state['update']
            torch.set_rng_state(state['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
            np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
            write_json(output/'resume.json', dict(checkpoint=str(Path(args.resume).resolve()),
                update=start_update, semantics='optimizer and RNG restored; simulator episodes restart'))
        if start_update >= cfg['updates']:
            raise ValueError('--updates is the total update target and must exceed the resume update')
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output/'tensorboard'))
        started = time.monotonic()
        last_update = start_update
        with (output/'metrics.jsonl').open('w', buffering=1) as log:
            for iteration in range(start_update, cfg['updates']):
                tick = time.monotonic()
                batches, rewards = {}, {}
                for r, pool in pools.items():
                    batches[r], rewards[r] = collect(pool, bank, policy, cfg['rollout_steps'])
                collection_seconds = time.monotonic()-tick
                metrics = update_policy(policy, optimizer, bank, batches, cfg)
                last_update = iteration+1
                frames = last_update*cfg['num_envs']*cfg['rollout_steps']*len(pools)
                metrics.update(update=last_update, transitions=frames, elapsed_seconds=time.monotonic()-started,
                               collection_seconds=collection_seconds, update_seconds=time.monotonic()-tick-collection_seconds,
                               transitions_per_second=cfg['num_envs']*cfg['rollout_steps']*len(pools)/(time.monotonic()-tick))
                for r, pool in pools.items():
                    metrics[f'{r}/mean_reward'] = rewards[r]
                    metrics[f'{r}/recoveries'] = pool.env.nonfinite_state_recovery_count
                    metrics[f'{r}/done_fraction'] = batches[r]['done'].float().mean().item()
                    for field in ('terminated','truncated','recovered'):
                        metrics[f'{r}/{field}_fraction'] = batches[r][field].float().mean().item()
                    for group in pool.env.reward_groups:
                        metrics[f'{r}/reward_{group}'] = batches[r][f'reward_{group}'].mean().item()
                    for s, count in zip(SOURCES, torch.bincount(batches[r]['source'], minlength=3).tolist()):
                        metrics[f'{r}/source_{s}_count'] = count
                        source_mask = batches[r]['source'] == SOURCES.index(s)
                        if count:
                            metrics[f'{r}/source_{s}_mean_reward'] = batches[r]['reward'][source_mask].mean().item()
                log.write(json.dumps(metrics)+'\n')
                for k,v in metrics.items():
                    writer.add_scalar(k, v, last_update)
                writer.flush()
                print(json.dumps(metrics), flush=True)
                if last_update % cfg['checkpoint_interval'] == 0:
                    checkpoint(output/f'checkpoint_{last_update}.pt', policy, optimizer, last_update, cfg, digest)
                if args.max_minutes and time.monotonic()-started >= args.max_minutes*60:
                    break
        checkpoint(output/'checkpoint_final.pt', policy, optimizer, last_update, cfg, digest)
        # Verify reconstruction reproduces all routes on the current observation batch.
        restored, _ = load_policy(output/'checkpoint_final.pt')
        with torch.no_grad():
            for r,pool in pools.items():
                obs=pool.observe(); commands=bank.gather(obs['motion'],obs['frame'])
                for s in range(3):
                    route=torch.full_like(obs['source'],s)
                    a=policy.actor(r,obs['proprio'],route,commands)[0].mean
                    b=restored.actor(r,obs['proprio'],route,commands)[0].mean
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
        write_json(output/'result.json', dict(status='complete', updates=last_update,
            requested_updates=cfg['updates'], budget_exhausted=last_update<cfg['updates'],
            checkpoint_roundtrip='passed_all_six_routes', payload_enabled=False,
            peak_torch_cuda_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started))
    finally:
        if writer is not None:
            writer.close()
        for pool in pools.values():
            pool.close()


def aggregate(rows):
    if not rows:
        return {}
    result = dict(clips=len(rows), families=len({r['family_id'] for r in rows}),
                  success_rate=float(np.mean([r['success'] for r in rows])),
                  completion=float(np.mean([r['completion'] for r in rows])),
                  recovery_clips=sum(r['reason']=='nonfinite' for r in rows))
    total=sum(r['steps'] for r in rows)
    for k in rows[0]['errors']:
        result[k] = sum(r['errors'][k]*r['steps'] for r in rows)/max(total,1)
    # Cluster resampling preserves dependence between related motion variants.
    families={f:[r for r in rows if r['family_id']==f] for f in sorted({r['family_id'] for r in rows})}
    sizes=np.asarray([len(v) for v in families.values()])
    successes=np.asarray([sum(r['success'] for r in v) for v in families.values()])
    indices=np.random.default_rng(42).integers(0,len(sizes),size=(1000,len(sizes)))
    boot=successes[indices].sum(1)/sizes[indices].sum(1)
    result['success_rate_family_bootstrap_ci95']=np.quantile(boot,[.025,.975]).tolist()
    return result


def evaluate(args):
    runtime_setup(); seed_all(args.seed)
    if args.num_envs < 1 or (args.max_motions is not None and args.max_motions < 1):
        raise ValueError("Evaluation environment and motion counts must be positive")
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    policy,state=load_policy(args.checkpoint); policy.eval()
    dataset=args.dataset or state['config']['dataset']
    if sha256(Path(dataset)/'dataset.json') != state['dataset_digest']:
        raise ValueError('Evaluation dataset must match training dataset manifest')
    bank=command_bank(dataset,args.split,'cuda:0',state['config'])
    count=min(len(bank.rows),args.max_motions or len(bank.rows))
    results=[]
    for robot in ROBOTS:
        seed_all(args.seed)
        pool=XWBCPool(robot,bank,min(args.num_envs,count),state['config']['history'],evaluation=True,
                      **history_options(state['config']))
        write_json(output/f'{robot}_environment.json',pool.resolved_config)
        try:
            for source_id,source in enumerate(SOURCES):
                pool.fixed_source=source_id
                for start in range(0,count,pool.num_envs):
                    real=min(pool.num_envs,count-start)
                    motion=torch.arange(start,start+pool.num_envs,device=bank.device).clamp_max(count-1)
                    pool.command.fixed_motion_ids=motion
                    seed_all(args.seed+start)
                    pool.reset(pool.all_ids)
                    active=torch.arange(pool.num_envs,device=bank.device)<real
                    sums={k:torch.zeros(pool.num_envs,device=bank.device) for k in pool.metrics()}
                    steps=torch.zeros(pool.num_envs,device=bank.device,dtype=torch.long)
                    while active.any():
                        with torch.no_grad():
                            obs=pool.observe(); commands=bank.gather(obs['motion'],obs['frame'])
                            dist,_=policy.actor(robot,obs['proprio'],obs['source'],commands)
                            transition,_=pool.step(dist.mean,dist.mean)
                            values=pool.metrics()
                            for k,v in values.items():
                                sums[k][active]+=v[active]
                            steps[active]+=1
                            complete=pool.command.t>=pool.command.lengths-1
                            recovered=pool.env._nonfinite_state_mask
                            finished=active & (transition['done'].squeeze(-1)|complete)
                            for i in finished.nonzero().flatten().tolist():
                                success=bool(complete[i] and not recovered[i])
                                row=bank.rows[int(motion[i])]
                                reasons=[k for k in pool.env.termination_funcs
                                         if float(transition['stats','termination',k][i])>0]
                                result=dict(motion_id=row['motion_id'],family_id=row['family_id'],category=row['category'],
                                    robot=robot,source=source,success=success,steps=int(steps[i]),
                                    completion=min(1.,int(steps[i])/max(1,int(pool.command.lengths[i])-1)),
                                    reason='complete' if success else ('nonfinite' if recovered[i] else ','.join(reasons)),
                                    errors={k:float(v[i]/steps[i]) for k,v in sums.items()})
                                results.append(result)
                                with (output/'clips.jsonl').open('a') as f:
                                    f.write(json.dumps(result,ensure_ascii=False)+'\n')
                            active[finished]=False
                print(f'evaluated {source} -> {robot}: {count} clips',flush=True)
        finally:
            pool.close()
    matrix={}
    for robot in ROBOTS:
        for source in SOURCES:
            rows=[r for r in results if r['robot']==robot and r['source']==source]
            cats={c:aggregate([r for r in rows if r['category']==c]) for c in sorted({r['category'] for r in rows})}
            matrix[f'{source}->{robot}']={**aggregate(rows),'categories':cats,
                'macro_category_success':float(np.mean([v['success_rate'] for v in cats.values()]))}
    write_json(output/'summary.json',dict(checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=sha256(args.checkpoint),split=args.split,seed=args.seed,matrix=matrix,
        dataset_digest=state['dataset_digest'],payload_enabled=False,
        initialization='target frame 0, fixed 2 cm lift, no pose/velocity noise',
        protocol='HEFT tracking terminations, complete-clip priority except nonfinite; errors include failed active steps'))
    with (output/'matrix.csv').open('w') as f:
        columns=['route','clips','families','success_rate','completion','mpkpe_world_m','mpkpe_root_relative_m','velocity_mps','root_rotation_rad','macro_category_success']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for route,m in matrix.items():
            writer.writerow(dict(route=route,**{k:m[k] for k in columns[1:]}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    tr=sub.add_parser('train')
    tr.add_argument('--config',default=str(ROOT/'cfg/xwbc/train.yaml'))
    tr.add_argument('--output',required=True)
    tr.add_argument('--dataset');tr.add_argument('--num-envs',type=int)
    tr.add_argument('--updates',type=int);tr.add_argument('--seed',type=int)
    tr.add_argument('--rollout-steps',type=int)
    tr.add_argument('--variant',choices=['shared','independent','no_align'])
    tr.add_argument('--resume');tr.add_argument('--max-minutes',type=float)
    ev=sub.add_parser('evaluate')
    ev.add_argument('--checkpoint',required=True);ev.add_argument('--output',required=True)
    ev.add_argument('--dataset');ev.add_argument('--split',choices=['val','test'],default='test')
    ev.add_argument('--num-envs',type=int,default=64);ev.add_argument('--seed',type=int,default=42)
    ev.add_argument('--max-motions',type=int)
    args=p.parse_args()
    train(args) if args.mode=='train' else evaluate(args)
