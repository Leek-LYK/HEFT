"""Stage D: critic warmup and conservative PPO from a verified stage-C policy."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from active_adaptation.envs.xwbc_staged import StagedPool
from active_adaptation.learning.modules.xwbc import ppo_loss
from active_adaptation.learning.modules.xwbc_staged import frozen_digest, action_mse
from active_adaptation.learning.xwbc import collect, history_options
from active_adaptation.learning.xwbc_staged import (
    config_for, runtime_setup, seed_all, make_bank, load_model, verify_gate,
    validate_parent, initialize_run, control_contract, checkpoint, log_metrics, write_json)
from active_adaptation.utils.xwbc_data import ROBOTS, SOURCES, sha256


class BalancedSourcePool(StagedPool):
    """Each environment slot keeps its source across resets: shares differ by <=1."""
    @torch.no_grad()
    def reset(self, ids):
        super().reset(ids)
        self.source[ids] = ids.remainder(len(SOURCES))


def validate_finetune_config(cfg):
    if cfg['num_envs'] < len(SOURCES):
        raise ValueError('D requires at least three environments per robot')
    warmup = cfg['critic_warmup_updates']
    if not isinstance(warmup, int) or warmup < 1 or cfg['updates'] <= warmup:
        raise ValueError('D needs positive critic warmup followed by PPO updates')
    if not 0 < cfg['action_std'] < 1:
        raise ValueError('D action_std must be between zero and one')
    for key in ('learning_rate', 'critic_learning_rate', 'target_kl', 'clip',
                'epochs', 'minibatch_size', 'max_grad_norm', 'value_weight'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(key + ' must be finite and positive')
    if not np.isfinite(cfg['anchor_weight']) or cfg['anchor_weight'] < 0:
        raise ValueError('anchor_weight must be finite and nonnegative')


def configure_finetune(model, cfg):
    """Frozen command encoders and fixed exploration; separate actor/critic groups."""
    model.requires_grad_(False)
    model.eval()
    actor = []
    for module in (model.proprio, model.backbones, model.heads):
        module.requires_grad_(True)
        actor.extend(module.parameters())
    model.critics.requires_grad_(True)
    critic = list(model.critics.parameters())
    with torch.no_grad():
        for parameter in model.log_std.values():
            parameter.fill_(math.log(cfg['action_std']))
    optimizer = torch.optim.Adam([
        dict(params=actor, lr=cfg['learning_rate'], name='actor'),
        dict(params=critic, lr=cfg['critic_learning_rate'], name='critic'),
    ])
    return optimizer, actor, critic


def actor_mean_digest(model):
    """All weights determining deterministic actions, including source encoders."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith(('critics.', 'log_std.')):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_finetune(args, model, optimizer, cfg, dataset_digest, parents):
    if not args.resume:
        return 0
    restored, state = load_model(args.resume, device=next(model.parameters()).device)
    if state['stage'] != 'D' or state['dataset_digest'] != dataset_digest:
        raise ValueError('D resume stage/data mismatch')
    if restored.constructor != model.constructor:
        raise ValueError('D resume model mismatch')
    if state['parents']['sha256'] != parents['sha256']:
        raise ValueError('D resume C anchor changed')
    # Only total budget and checkpoint cadence may change on a true resume.
    compatible = lambda c: {k: v for k, v in c.items() if k not in ('updates', 'checkpoint_interval')}
    if compatible(state['config']) != compatible(cfg):
        raise ValueError('D resume configuration changed; use a separate experiment')
    model.load_state_dict(restored.state_dict())
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    return state['update']


def fixed_std_kl(mean, old_mean, std):
    """Exact KL for diagonal Gaussians with the same fixed exploration scale."""
    return .5 * ((mean-old_mean)/std).square().sum(-1).mean()


@torch.no_grad()
def bounded_optimizer_step(model, optimizer, actor_params, probes, cfg):
    """Backtrack the actor's Adam parameter delta; keep its gradient moments.

    Accepted interpolation is an effective actor learning-rate reduction for this
    step. A fully rejected actor step also restores actor optimizer state. Critic
    updates are independent and remain applied.
    """
    before = [p.detach().clone() for p in actor_params]
    old_state = {p: ({k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()}
                     if p in optimizer.state else None) for p in actor_params}
    optimizer.step()
    proposed = [p.detach().clone() for p in actor_params]

    def divergence():
        values = []
        for robot, probe in probes.items():
            dist, _ = model.actor(robot, probe['proprio'], probe['source'], probe['commands'])
            values.append(float(fixed_std_kl(dist.mean,probe['old_mean'],dist.stddev)))
        return max(values,default=0.)

    maximum = cfg.get('max_backtracks',10)
    proposed_kl = divergence()
    for attempt in range(maximum+1):
        scale = .5**attempt
        if attempt:
            for parameter, original, new in zip(actor_params,before,proposed):
                parameter.copy_(original+scale*(new-original))
        kl = proposed_kl if attempt==0 else divergence()
        if np.isfinite(kl) and kl <= cfg['target_kl']:
            return dict(actor_step_scale=scale,backtracks=attempt,actor_step_accepted=1,
                        post_step_kl=kl,max_proposed_kl=proposed_kl)
    for parameter, original in zip(actor_params,before):
        parameter.copy_(original)
        if old_state[parameter] is None:
            optimizer.state.pop(parameter,None)
        else:
            optimizer.state[parameter] = old_state[parameter]
    return dict(actor_step_scale=0.,backtracks=maximum+1,actor_step_accepted=0,
                post_step_kl=divergence(),max_proposed_kl=proposed_kl)


def update_finetune(model, anchor, optimizer, actor_params, critic_params,
                    bank, batches, cfg, warmup):
    """Warmup has no actor backward; PPO averages target losses after native heads."""
    metrics = []
    n = len(next(iter(batches.values()))['value'])
    stopped = False
    observed_kl = 0.
    for _ in range(cfg['epochs']):
        permutations = {r: torch.randperm(n, device=bank.device) for r in batches}
        for start in range(0, n, cfg['minibatch_size']):
            optimizer.zero_grad(set_to_none=True)
            losses, row, batch_kls, probes = [], {}, [], {}
            for robot, batch in batches.items():
                ids = permutations[robot][start:start + cfg['minibatch_size']]
                b = {k: v[ids] for k, v in batch.items()}
                value = model.value(robot, b['proprio'], b['privileged'])
                value_loss = (value - b['returns']).square().mean()
                pg = value_loss.new_zeros(())
                entropy = pg.clone(); kl = pg.clone(); keep = pg.clone()
                loss = cfg['value_weight'] * value_loss
                if not warmup:
                    commands = bank.gather(b['motion'], b['frame'])
                    distribution, _ = model.actor(robot, b['proprio'], b['source'], commands)
                    pg, entropy, kl = ppo_loss(distribution, b['action'], b['logp'], b['advantage'], cfg['clip'])
                    with torch.no_grad():
                        baseline, _ = anchor.actor(robot, b['proprio'], b['source'], commands)
                    keep = action_mse(distribution.mean, baseline.mean)
                    loss = loss + pg + cfg['anchor_weight'] * keep
                    batch_kls.append(float(fixed_std_kl(distribution.mean,b['old_mean'],distribution.stddev).detach()))
                    probes[robot]=dict(proprio=b['proprio'],source=b['source'],commands=commands,old_mean=b['old_mean'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(robot + ': nonfinite D loss')
                losses.append(loss / len(batches))
                row.update({robot + '/' + key: float(v.detach()) for key, v in
                            dict(policy_loss=pg, value_loss=value_loss, entropy=entropy,
                                 kl=kl, anchor_action_mse=keep).items()})
            observed_kl = max(observed_kl, max(batch_kls, default=0.))
            if not warmup and max(batch_kls, default=0.) > cfg['target_kl']:
                stopped = True
                break
            sum(losses).backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(actor_params, cfg['max_grad_norm'], error_if_nonfinite=True)
            critic_norm = torch.nn.utils.clip_grad_norm_(critic_params, cfg['max_grad_norm'], error_if_nonfinite=True)
            if warmup:
                optimizer.step()
                step_metrics=dict(actor_step_scale=0.,backtracks=0,actor_step_accepted=0,post_step_kl=0.,max_proposed_kl=0.)
            else:
                step_metrics=bounded_optimizer_step(model,optimizer,actor_params,probes,cfg)
            row.update(step_metrics)
            row.update(actor_gradient_norm=float(actor_norm), critic_gradient_norm=float(critic_norm))
            metrics.append(row)
            if not warmup and not step_metrics['actor_step_accepted']:
                stopped=True
                break
        if stopped:
            break
    if not metrics:
        raise RuntimeError('D optimizer made no update; inspect initial policy KL')
    result = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    result.update(optimizer_steps=len(metrics), actor_optimizer_steps=sum(m['actor_step_accepted'] for m in metrics),
                  kl_early_stop=int(stopped), max_observed_kl=observed_kl,
                  max_post_step_kl=max(m['post_step_kl'] for m in metrics),
                  max_proposed_kl=max(m['max_proposed_kl'] for m in metrics))
    with torch.no_grad():
        for robot, b in batches.items():
            predicted = model.value(robot, b['proprio'], b['privileged'])
            variance = b['returns'].var(unbiased=False)
            result[robot + '/explained_variance'] = float(
                1 - (b['returns'] - predicted).var(unbiased=False) / variance.clamp_min(1e-8))
            result[robot + '/return_mean'] = float(b['returns'].mean())
            result[robot + '/action_std'] = float(model.log_std[robot].exp().mean())
    return result


def rollout_metrics(batches, pools):
    result = {}
    for robot, b in batches.items():
        result[robot + '/mean_reward'] = float(b['reward'].mean())
        result[robot + '/recoveries'] = pools[robot].env.nonfinite_state_recovery_count
        for field in ('terminated', 'truncated', 'recovered'):
            result[robot + '/' + field + '_fraction'] = float(b[field].float().mean())
        for name in pools[robot].env.reward_groups:
            result[robot + '/reward_' + name] = float(b['reward_' + name].mean())
        for index, source in enumerate(SOURCES):
            mask = b['source'] == index
            if not mask.any():
                raise RuntimeError('D rollout missing a source: ' + robot + '/' + source)
            prefix = robot + '/' + source + '/'
            done = int(b['done'][mask].sum())
            success = int((b['completed'][mask] & ~b['recovered'][mask]).sum())
            result.update({prefix + 'transitions': int(mask.sum()), prefix + 'mean_reward': float(b['reward'][mask].mean()),
                           prefix + 'ended_episodes': done, prefix + 'completed_episodes': success,
                           prefix + 'train_episode_success_rate': success / max(done, 1)})
    return result


def train_finetune(args):
    runtime_setup()
    cfg = config_for(args.config, 'D', args)
    validate_finetune_config(cfg)
    seed_all(cfg['seed'])
    summary = verify_gate(args.policy_gate, args.policy, cfg, 'C')
    anchor, parent = load_model(args.policy)
    validate_parent(parent, cfg, 'C')
    anchor.requires_grad_(False); anchor.eval()
    model, _ = load_model(args.policy)
    initial_actor = actor_mean_digest(model)
    optimizer, actor_params, critic_params = configure_finetune(model, cfg)
    parents = dict(policy=str(Path(args.policy).resolve()), sha256=sha256(args.policy),
                   gate=str(Path(args.policy_gate).resolve()), baselines=summary['matrix'], controls=parent['controls'])
    digest = sha256(Path(cfg['dataset']) / 'dataset.json')
    fixed = frozen_digest(model)
    start = restore_finetune(args, model, optimizer, cfg, digest, parents)
    if frozen_digest(model) != fixed:
        raise ValueError('D resume modified frozen encoders or exploration scale')
    if start >= cfg['updates']:
        raise ValueError('D update target must exceed resume update')
    out, writer = initialize_run(args.output, cfg)
    pools = {}
    try:
        bank = make_bank(cfg, 'train')
        controls = {}
        for robot in ROBOTS:
            pool = BalancedSourcePool(robot, bank, cfg['num_envs'], cfg['history'], **history_options(cfg))
            pools[robot] = pool
            controls[robot] = control_contract(pool)
            if controls[robot] != parents['controls'][robot]:
                raise ValueError('D action contract differs from C: ' + robot)
            write_json(out / (robot + '_environment.json'), pool.resolved_config)
        write_json(out / 'model.json', model.constructor)
        write_json(out / 'control_contract.json', controls)
        write_json(out / 'trainable_parameters.json', [n for n, p in model.named_parameters() if p.requires_grad])
        write_json(out / 'frozen_modules.json', dict(sha256=fixed, stage='D', meaning='command encoders and fixed log_std'))
        write_json(out / 'initialization.json', dict(parent_C=parents['policy'], parent_sha256=parents['sha256'],
                   actor_mean_sha256=initial_actor, exploration_std=cfg['action_std'], critic_warmup_updates=cfg['critic_warmup_updates']))
        if args.resume:
            write_json(out / 'resume.json', dict(checkpoint=str(Path(args.resume).resolve()), update=start,
                       semantics='model/optimizer/RNG restored; simulator episodes restart'))
        started = time.monotonic(); last = start
        with (out / 'metrics.jsonl').open('w', buffering=1) as log:
            for iteration in range(start, cfg['updates']):
                tick = time.monotonic()
                warmup = iteration < cfg['critic_warmup_updates']
                batches = {robot: collect(pool, bank, model, cfg['rollout_steps'])[0] for robot, pool in pools.items()}
                collection_seconds = time.monotonic() - tick
                m = update_finetune(model, anchor, optimizer, actor_params, critic_params, bank, batches, cfg, warmup)
                m.update(rollout_metrics(batches, pools))
                last = iteration + 1
                m.update(update=last, transitions=last * cfg['num_envs'] * cfg['rollout_steps'] * len(pools),
                         critic_warmup=int(warmup), ppo_update=max(0, last-cfg['critic_warmup_updates']),
                         elapsed_seconds=time.monotonic()-started, collection_seconds=collection_seconds,
                         iteration_seconds=time.monotonic()-tick)
                log_metrics(log, writer, m)
                if last <= cfg['critic_warmup_updates'] and (last == 1 or last % cfg['checkpoint_interval'] == 0 or last == cfg['critic_warmup_updates']):
                    if actor_mean_digest(model) != initial_actor:
                        raise RuntimeError('D actor changed during critic-only warmup')
                if last == cfg['critic_warmup_updates']:
                    write_json(out / 'critic_warmup_complete.json', dict(update=last, actor_unchanged=True,
                               metrics={k:v for k,v in m.items() if 'value_loss' in k or 'explained_variance' in k}))
                if last % cfg['checkpoint_interval'] == 0 or last == cfg['critic_warmup_updates']:
                    if frozen_digest(model) != fixed:
                        raise RuntimeError('D frozen command encoders or exploration changed')
                    checkpoint(out / f'checkpoint_{last}.pt', model, optimizer, 'D', last, cfg, digest, parents, controls)
                if args.max_minutes and time.monotonic()-started >= args.max_minutes*60:
                    break
        if frozen_digest(model) != fixed:
            raise RuntimeError('D frozen modules changed')
        checkpoint(out / 'checkpoint_final.pt', model, optimizer, 'D', last, cfg, digest, parents, controls)
        write_json(out / 'result.json', dict(status='complete', stage='D', updates=last, requested_updates=cfg['updates'],
                   budget_exhausted=last<cfg['updates'], frozen_modules='unchanged', critic_warmup_updates=min(last,cfg['critic_warmup_updates']),
                   ppo_updates=max(0,last-cfg['critic_warmup_updates']), elapsed_seconds=time.monotonic()-started))
    finally:
        writer.close()
        for pool in pools.values():
            pool.close()
