"""On-policy six-source teacher correction, initialized from an approved B or C student."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.learning.xwbc_staged import (
    config_for, runtime_setup, seed_all, load_model, validate_parent, verify_gate,
    initialize_run, make_bank, control_contract, query_teacher, checkpoint,
    log_metrics, write_json)
from active_adaptation.learning.xwbc import history_options
from active_adaptation.learning.xwbc_finetune import BalancedSourcePool
from active_adaptation.learning.modules.xwbc_staged import Reservoir, student_mean, frozen_digest
from active_adaptation.utils.xwbc_data import ROBOTS, SOURCES, ROOT, sha256


def configure(model, train_sources=False):
    model.requires_grad_(False)
    for module in (model.proprio, model.backbones, model.heads):
        module.requires_grad_(True)
    if train_sources:
        model.encoders.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def collect(model, teacher, pool, bank, replay, steps):
    reward = error = 0.
    robot = pool.robot_name
    for _ in range(steps):
        obs = pool.observe()
        commands = bank.gather(obs['motion'], obs['frame'])
        target = query_teacher(teacher, pool, obs, commands)
        means = torch.stack([student_mean(model, robot, obs['proprio'], commands[s], s) for s in SOURCES], 1)
        action = means[torch.arange(pool.num_envs, device=bank.device), obs['source']]
        replay.add(dict(proprio=obs['proprio'], motion=obs['motion'], frame=obs['frame'], target=target))
        error += float((action-target).square().mean())
        transition, _ = pool.step(action, action)
        reward += float(transition['reward'].sum(-1).mean())
        pool.reset(transition['done'].squeeze(-1).nonzero().flatten())
    return {'reward': reward/steps, 'visited_teacher_mse': error/steps}


def update(model, anchor, optimizer, bank, replays, cfg):
    totals = {}
    for _ in range(cfg['supervised_steps']):
        optimizer.zero_grad(set_to_none=True)
        for robot, replay in replays.items():
            batch = replay.sample(cfg['batch_size'], bank.device)
            commands = bank.gather(batch['motion'], batch['frame'])
            for source in SOURCES:
                pred = student_mean(model, robot, batch['proprio'], commands[source], source)
                teacher_loss = (pred-batch['target']).square().mean()
                if cfg['anchor_weight']:
                    if anchor is None:
                        raise ValueError('Nonzero anchor weight requires an anchor')
                    with torch.no_grad():
                        old = student_mean(anchor, robot, batch['proprio'], commands[source], source)
                    anchor_loss = (pred-old).square().mean()
                else:
                    anchor_loss = pred.new_zeros(())
                ((teacher_loss+cfg['anchor_weight']*anchor_loss)/(len(replays)*len(SOURCES))).backward()
                for key, value in [('teacher_mse',teacher_loss), ('anchor_mse',anchor_loss)]:
                    name = robot+'/'+source+'/'+key
                    totals[name] = totals.get(name,0.)+float(value.detach())/cfg['supervised_steps']
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
    totals['gradient_norm'] = float(grad)
    return totals


def iteration_config(cfg, iteration):
    """Budget schedule changes sampling/optimization counts, never the E objective."""
    result = dict(cfg)
    schedule = cfg.get('budget_schedule', [])
    previous = 0
    for block in schedule:
        if block['until_update'] <= previous or min(block['supervised_steps'], block['batch_size']) <= 0:
            raise ValueError('Invalid budget schedule')
        previous = block['until_update']
    if schedule and previous < cfg['updates']:
        raise ValueError('Budget schedule does not cover training')
    for block in schedule:
        if iteration <= block['until_update']:
            result.update({k:block[k] for k in ('supervised_steps','batch_size')})
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'cfg/xwbc/v02/teacher_refinement.yaml'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-root', default=str(ROOT/'logs/v0.2/staged_run01'))
    parser.add_argument('--updates', type=int)
    parser.add_argument('--num-envs', type=int)
    args = parser.parse_args()
    runtime_setup()
    cfg = config_for(args.config, 'E', args)
    seed_all(cfg['seed'])
    root = Path(args.baseline_root)
    parent_stage = cfg.get('parent_stage', 'C')
    if parent_stage not in ('B', 'C'):
        raise ValueError('Expected B or C initialization')
    if parent_stage == 'B' and not cfg['train_sources']:
        raise ValueError('B initialization requires train_sources')
    parent_dir = 'student_human' if parent_stage == 'B' else 'source_encoders'
    gate_dir = cfg.get('parent_gate_dir', 'eval_val_capability' if parent_stage == 'B' else 'eval_val')
    cpath = root/parent_dir/'checkpoint_final.pt'
    summary = verify_gate(root/parent_dir/gate_dir/'gate.json', cpath, cfg, parent_stage)
    model, state = load_model(cpath)
    validate_parent(state, cfg, parent_stage)
    reference = None
    if parent_stage == 'B':
        # Evaluation-only scores: no C/E weights or outputs enter optimization.
        directory = ROOT/cfg['evaluation_reference']
        reference = json.loads((directory/'summary.json').read_text())
        gate = json.loads((directory/'gate.json').read_text())
        if sha256(directory/'summary.json') != gate['summary_sha256']:
            raise ValueError('Reference summary changed')
        for key in ('dataset_digest', 'context', 'protocol', 'motion_ids_digest'):
            if reference[key] != summary[key]:
                raise ValueError('Evaluation reference mismatch: '+key)
        if reference['split'] != 'val' or not reference['full_scope'] or set(reference['matrix']) != {s+'->'+r for r in ROBOTS for s in SOURCES}:
            raise ValueError('Full six-route validation reference required')
    iteration_config(cfg, 1)
    anchor = copy.deepcopy(model).requires_grad_(False)
    params = configure(model, cfg['train_sources'])
    frozen = frozen_digest(model)
    teachers = {}
    teacher_states = {}
    for robot in ROBOTS:
        path = root/f'teacher_{robot}/checkpoint_final.pt'
        teachers[robot], teacher_states[robot] = load_model(path)
        validate_parent(teacher_states[robot], cfg, 'A')
        if teachers[robot].robot != robot:
            raise ValueError('Teacher robot mismatch')
        teachers[robot].requires_grad_(False)
    parents = dict(checkpoint=str(cpath), sha256=sha256(cpath), baselines=summary['matrix'],
                   teachers={r:dict(checkpoint=str(root/f'teacher_{r}/checkpoint_final.pt'),
                                    sha256=sha256(root/f'teacher_{r}/checkpoint_final.pt')) for r in ROBOTS})
    parents['initialization_stage'] = parent_stage
    parents['anchor'] = dict(checkpoint=str(cpath), sha256=parents['sha256'], mode='same_source_initial_student')
    if reference is not None:
        parents['baselines'] = reference['matrix']
        parents['evaluation_reference'] = dict(directory=str(directory), checkpoint_sha256=reference['checkpoint_sha256'],
                                               summary_sha256=sha256(directory/'summary.json'), role='evaluation_only')
    out, writer = initialize_run(args.output, cfg)
    extra = ['scripts/refine_xwbc_student.py', 'cfg/xwbc/v02/teacher_refinement.yaml', 'cfg/xwbc/v02/teacher_refinement_all_sources.yaml', 'scripts/run_xwbc_no_c_ablation.py', 'scripts/compare_xwbc_refinement.py', 'scripts/audit_xwbc_refinement.py']
    actual_config = Path(args.config).resolve().relative_to(ROOT)
    if str(actual_config) not in extra:
        extra.append(str(actual_config))
    hashes = json.loads((out/'source_hashes.json').read_text())
    for relative in extra:
        destination = out/'code'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
        hashes[relative] = sha256(ROOT/relative)
    write_json(out/'source_hashes.json', hashes)
    write_json(out/'parents.json', parents)
    write_json(out/'trainable_parameters.json', [n for n,p in model.named_parameters() if p.requires_grad])
    optimizer = torch.optim.Adam(params, lr=cfg['learning_rate'])
    bank = make_bank(cfg, 'train')
    replay = {r:Reservoir(cfg['replay_capacity'], cfg['seed']+i) for i,r in enumerate(ROBOTS)}
    pools = {}
    started = time.monotonic()
    optimizer_steps = replay_samples = 0
    try:
        for robot in ROBOTS:
            pool = BalancedSourcePool(robot, bank, cfg['num_envs'], cfg['history'], **history_options(cfg))
            pools[robot] = pool
            contract = control_contract(pool)
            if contract != state['controls'][robot] or contract != teacher_states[robot]['controls'][robot]:
                raise ValueError('Student/teacher control contract mismatch')
            write_json(out/f'{robot}_environment.json', pool.resolved_config)
        controls = {r:control_contract(p) for r,p in pools.items()}
        with (out/'metrics.jsonl').open('w', buffering=1) as log:
            for iteration in range(1,cfg['updates']+1):
                row = {}
                for robot,pool in pools.items():
                    stats = collect(model,teachers[robot],pool,bank,replay[robot],cfg['rollout_steps'])
                    row.update({robot+'/'+k:v for k,v in stats.items()})
                    row[robot+'/recoveries'] = pool.env.nonfinite_state_recovery_count
                active_cfg = iteration_config(cfg, iteration)
                row.update(update(model,anchor,optimizer,bank,replay,active_cfg))
                optimizer_steps += active_cfg['supervised_steps']
                replay_samples += active_cfg['supervised_steps']*active_cfg['batch_size']*len(ROBOTS)
                row.update(optimizer_steps=optimizer_steps, replay_samples=replay_samples,
                           supervised_steps=active_cfg['supervised_steps'], batch_size=active_cfg['batch_size'])
                row.update(update=iteration,transitions=iteration*cfg['num_envs']*cfg['rollout_steps']*len(ROBOTS),elapsed_seconds=time.monotonic()-started)
                log_metrics(log,writer,row)
                if iteration%cfg['checkpoint_interval']==0 or iteration==cfg['updates']:
                    if frozen_digest(model)!=frozen:
                        raise RuntimeError('Frozen source/critic parameters changed')
                    checkpoint(out/f'checkpoint_{iteration}.pt',model,optimizer,'E',iteration,cfg,state['dataset_digest'],parents,controls,replay)
        checkpoint(out/'checkpoint_final.pt',model,optimizer,'E',iteration,cfg,state['dataset_digest'],parents,controls,replay)
        write_json(out/'result.json',dict(status='complete',stage='E_T' if cfg.get('transformer') else 'E',method=cfg['method'],updates=iteration,
                                        optimizer_steps=optimizer_steps,replay_samples=replay_samples,ppo_updates=0,teacher_takeovers=0,frozen_modules='unchanged',elapsed_seconds=time.monotonic()-started))
    finally:
        writer.close()
        for pool in pools.values():
            pool.close()


if __name__ == '__main__':
    main()
