"""Read-only observation/physics assertions on a live smoke environment."""
import json
from pathlib import Path
import torch

def probe(env,policy,carry,output,condition):
    base=env.base_env;cmd=base.command_manager
    assert not cmd.student_motion_randomization_cfg['enable']
    assert not base.randomizations['window_cap_hand_load'].enabled
    clean,raw=cmd.dataset.get_teacher_student_slice(None,cmd.t,teacher_steps=cmd.student_future_steps,student_steps=cmd.student_future_steps)
    assert torch.equal(clean.root_pos_w,raw.root_pos_w)
    assert torch.equal(clean.root_quat_w,raw.root_quat_w)
    delta=(clean.joint_pos-raw.joint_pos).abs()
    if condition=='C0':assert torch.count_nonzero(delta)==0
    else:assert torch.count_nonzero(delta)>0,'No corruption observed in sampled C1 windows'
    from torchrl.envs.utils import set_exploration_type,ExplorationType
    with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
        a=carry.clone();b=carry.clone();b['priv']=torch.ones_like(b['priv']);b['priv_critic']=torch.ones_like(b['priv_critic'])
        def student(td):
            policy.adapt_module(td);policy.student_latent_quantizer(td);policy.actor_student(td);return td['loc']
        assert torch.equal(student(a),student(b))
    tensors={}
    for name,obj in base.randomizations.items():
        tensors[name]={}
        for key,value in vars(obj).items():
            if isinstance(value,torch.Tensor) and value.numel() and value.dtype.is_floating_point:
                tensors[name][key]=dict(shape=list(value.shape),min=float(value.min()),max=float(value.max()))
    action=base.action_manager
    report=dict(status='PASS',condition=condition,legacy_transform=False,hand_load=False,root_equal=True,paired_joint_max_delta=float(delta.max()),paired_changed_elements=int(torch.count_nonzero(delta)),student_clean_priv_invariance=True,physics_objects=tensors,action_offset_span=float(action.offset.max()-action.offset.min()))
    assert report['action_offset_span']>0,'joint offset DR unexpectedly inactive'
    Path(output,'runtime_probe.json').write_text(json.dumps(report,indent=2)+'\n')
