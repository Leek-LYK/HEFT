"""Opt-in research contract checks; legacy runs are unchanged."""
import json
import os
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

def check_config(cfg):
    if not cfg.get('refinetrack',{}).get('enabled',False):return
    assert cfg.task.num_envs==4096
    assert not cfg.task.command.student_motion_randomization.enable
    assert not cfg.task.randomization.window_cap_hand_load.enabled
    assert not cfg.task.command.dataset_extra_keys
    assert not cfg.task.action.boot_protect, 'clean boot override leaks clean target into control'
    expected={'perturb_body_com','perturb_body_materials','motor_params_implicit','random_joint_offset','perturb_root_vel','perturb_body_wrench','perturb_gravity'}
    assert expected <= set(cfg.task.randomization), set(cfg.task.randomization)
    assert not any('load' in str(k).lower() for k in cfg.task.observation)
    condition=cfg.refinetrack.condition
    groups=cfg.task.command.dataset.groups
    if condition=='C0':assert all(g.teacher_mem_path==g.student_mem_path for g in groups)
    elif condition=='C1':assert any(g.teacher_mem_path!=g.student_mem_path for g in groups)
    else:raise ValueError(condition)
    print('REFINETRACK_CONTRACT '+json.dumps(dict(condition=condition,reference_corruption_source='none' if condition=='C0' else 'offline_paired_joint_v1_r1',student_motion_transform=False,physics_DR=OmegaConf.to_container(cfg.task.randomization,resolve=True),action_DR=OmegaConf.to_container(cfg.task.action,resolve=True),hand_load_label=False,teacher_encoder='clean privileged',student='raw policy only',root_corruption=False)),flush=True)

def verify_loaded(policy, state):
    # Check shapes/keys before the native loader's warning-only path.
    for name,module in policy.named_children():
        actual=policy._unwrap_module(module).state_dict();saved=state.get(name,{})
        if set(actual)!=set(saved):raise ValueError(f'checkpoint keys mismatch: {name}')
        for k,v in actual.items():
            if v.shape!=saved[k].shape:raise ValueError(f'checkpoint shape mismatch: {name}.{k}')

def log_iteration(output_dir,i,rollout_s,training_s,transitions):
    import torch
    torch.cuda.synchronize()
    row=dict(iteration=i,rollout_s=rollout_s,training_s=training_s,transitions=transitions,rollout_transitions_per_s=transitions/rollout_s,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    row['nvidia_smi_used_mib']=int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    with (Path(output_dir)/'refinetrack_iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
