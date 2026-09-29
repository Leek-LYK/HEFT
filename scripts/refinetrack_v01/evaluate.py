"""Fixed-case, pre-reset metrics. Test is never a checkpoint-selection source."""
import argparse
import copy
import json
import os
import sys
import types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType,set_exploration_type
from scripts.refinetrack_v01.build_data import ROOT,sha,dump,stable
from scripts.refinetrack_v01.preflight import check_pair
from scripts.utils.helpers import make_env_policy
from scripts.refinetrack_v01.metrics import completion_result
from active_adaptation.utils.math import quat_apply_inverse,quat_in_frame

def angle(a,b):
    a=a/a.norm(dim=-1,keepdim=True).clamp_min(1e-8);b=b/b.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    return 2*torch.acos((a*b).sum(-1).abs().clamp(0,1))

def nominal(cfg):
    r=cfg.task.randomization
    r.perturb_body_com.com_range=[0.,0.]
    for key in ['stiffness_range','damping_range','armature_range']:
        for name in r.motor_params_implicit[key]:r.motor_params_implicit[key][name]=[1.,1.]
    for name in r.random_joint_offset:r.random_joint_offset[name]=[0.,0.]
    for name in ['x','y','z','roll','pitch','yaw']:r.perturb_root_vel[name]=[0.,0.]
    r.perturb_body_wrench.force_magnitude_range_n=[0.,0.]
    r.perturb_gravity.std=0.
    r.perturb_body_materials.static_friction_range=[.6,.6]
    r.perturb_body_materials.solref_time_constant_range=[.02,.02]
    r.perturb_body_materials.solref_dampratio_range=[1.,1.]
    r.window_cap_hand_load.enabled=False
    cfg.task.action.alpha=[.9,.9];cfg.task.action.max_delay=0
    cfg.task.action.torque_limit_scale_range=[1.,1.];cfg.task.action.boot_protect=False
    for name in cfg.task.command.init_noise:cfg.task.command.init_noise[name]=0.
    for group in cfg.task.observation.values():
        for provider in group.values():
            if isinstance(provider,dict) or OmegaConf.is_dict(provider):
                for key in list(provider):
                    if 'noise_std' in key:provider[key]=0.
    for group in ['shared_joint_pos','shared_joint_vel']:
        for key,value in cfg.task.command[group].items():
            if OmegaConf.is_dict(value):
                for name in value:value[name]=0.
            else:cfg.task.command[group][key]=0.

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,default=ROOT/'dataset/roban/refinetrack_v01/r2');p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--severity',choices=['S0','S1','S2','S3'],default='S0');p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--seed',type=int,default=0);p.add_argument('--role',choices=['teacher','student'],default='student');p.add_argument('--recipe',choices=['trained','heldout'],default='trained');p.add_argument('--legacy-clean',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['3','4','7']
    a.output.mkdir(parents=True,exist_ok=False)
    from scripts.refinetrack_v01.preflight import check_data
    check_data(a.data)
    if a.legacy_clean:assert a.severity=='S0' and a.split=='validation'
    clean=a.data/a.split/('legacy_clean' if a.legacy_clean else 'clean');raw=clean if a.severity=='S0' else a.data/a.split/(f'raw_{a.severity}' if a.recipe=='trained' else f'raw_heldout_{a.severity}')
    check_pair(clean,raw)
    labels=json.loads((clean/'id_label.json').read_text());ids=sorted(range(len(labels)),key=lambda i:stable(json.dumps(labels[i],sort_keys=True)))
    if a.limit:ids=ids[:a.limit]
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=OmegaConf.create(copy.deepcopy(OmegaConf.to_container(checkpoint['cfg'],resolve=True)))
    OmegaConf.set_struct(cfg,False);cfg.checkpoint_path=str(a.checkpoint.resolve());cfg.seed=a.seed;cfg.vecnorm='eval';cfg.task.num_envs=len(ids)
    cfg.algo.phase='train' if a.role=='teacher' else 'finetune';cfg.task.student_train=True
    cfg.task.command.dataset.groups=[dict(name='evaluation',teacher_mem_path=str(clean.resolve()),student_mem_path=str(raw.resolve()),weight=1.)]
    cfg.task.command.dataset.resample_on_reset=False;cfg.task.command.student_motion_randomization.enable=False
    cfg.task.max_episode_length=100000;cfg.task.termination.motion_xy_range_termination.motion_xy_max_offset=None
    cfg.headless=True;cfg.app.headless=True;cfg.app.enable_cameras=False;cfg.eval_render=False;cfg.wandb.mode='disabled';cfg.train_record.enabled=False
    cfg.refinetrack={'enabled':True,'condition':'evaluation'}
    if 'pilot' in cfg:cfg.pilot.enabled=False
    if 'smoke' in cfg:cfg.smoke.enabled=False
    nominal(cfg);OmegaConf.save(cfg,a.output/'resolved_config.yaml')
    env,policy,vecnorm,_=make_env_policy(cfg);base=env.base_env;cmd=base.command_manager;ds=cmd.dataset;device=base.device
    env.eval();policy.eval();env.step_schedule(1.,-1)
    for name in ['teacher','student','critic','estimator']:
        def forbidden(*args,**kwargs):raise AssertionError('evaluation optimizer step')
        getattr(policy,'opt_'+name).step=forbidden
    body_names=[name for name in ds.body_names if name!='base_link']
    body_m=torch.tensor([ds.body_names.index(name) for name in body_names],device=device)
    body_a=torch.tensor([base.scene['robot'].body_names.index(name) for name in body_names],device=device)
    from active_adaptation.envs.mdp.contact_utils import resolve_contact_indices
    from active_adaptation.envs.mdp.rewards.locomotion import _get_contact_majority_cache
    sensor=base.scene['contact_forces'];feet_ids,_,_=resolve_contact_indices(sensor,base.scene['robot'],list(cmd.feet_names_asset))
    feet_ids=torch.tensor(feet_ids,device=device);contact_cache=_get_contact_majority_cache(base,sensor)
    fixed=torch.tensor(ids,device=device,dtype=torch.long)
    def fixed_init(self,env_ids):
        ds._motion_ids_A[env_ids]=fixed[env_ids];ds._dataset_ids_A[env_ids]=0;ds._len_A[env_ids]=ds.global_lengths[fixed[env_ids]].to(torch.int32);ds._invalidate_cache(env_ids)
        self.t[env_ids]=0;self.lengths[env_ids]=ds._len_A[env_ids];self._reinit_requested[env_ids]=False
        t,r=ds.get_teacher_student_slice(env_ids,self.t[env_ids],teacher_steps=1,student_steps=1)
        self._set_motion_origin_offset(env_ids,t,r);t,_=self._apply_motion_origin_offset(t,r,env_ids)
        self.sample_init_robot(env_ids,t)
    cmd.sample_init=types.MethodType(fixed_init,cmd)
    carry=env.reset();rollout=policy.get_rollout_policy();n=len(ids);active=torch.ones(n,dtype=torch.bool,device=device)
    lengths=ds.global_lengths[fixed].cpu().numpy();case=[dict(motion_id=i,source=labels[i],planned_transitions=int(lengths[k]-1),frames=0,success=False,termination_reasons=[],metrics={}) for k,i in enumerate(ids)]
    totals={};qtrace=[[] for _ in ids];rtrace=[[] for _ in ids];atrace=[[] for _ in ids];roottrace=[[] for _ in ids];refroottrace=[[] for _ in ids];prev=torch.zeros(n,21,device=device);quant_hist=torch.zeros(9,device=device,dtype=torch.long)
    no_leak=True;paired_max=0.;root_max=0.
    with torch.inference_mode(),set_exploration_type(ExplorationType.MODE):
        for tick in range(int(lengths.max())+1):
            before=carry.clone();carry=rollout(carry)
            if tick==1:
                probe=before.clone();probe['priv']=torch.ones_like(probe['priv']);probe['priv_critic']=torch.ones_like(probe['priv_critic'])
                if a.role=='student':no_leak=torch.equal(rollout(probe)['action'],carry['action']);assert no_leak
            d=before.clone();policy.encoder_priv(d);policy.adapt_module(d);policy.student_latent_quantizer(d)
            zt=d['priv_feature'];zs=d['priv_pred'];qz=d['student_priv_feature'];quant_hist+=torch.bincount((qz[active].long()+4).flatten(),minlength=9)
            policy.actor_teacher(d);ta=d['loc'].clone();policy.actor_student(d);sa=d['loc'].clone();d['student_priv_feature']=zs;policy.actor_student(d);unquant=d['loc'].clone()
            action=carry['action'].clone();transition=env.step(carry);asset=base.scene['robot'].data;m=cmd._motion
            actual=asset.joint_pos[:,cmd.joint_idx_asset];ref=m.joint_pos[:,0,cmd.joint_idx_motion]
            ar=asset.root_link_quat_w;rr=m.root_quat_w[:,0]
            root=asset.root_link_pos_w-base.scene.env_origins;refroot=m.root_pos_w[:,0]
            metrics=dict(joint_rmse_rad=((actual-ref)**2).mean(-1).sqrt(),root_position_m=(root-refroot).norm(dim=-1),root_orientation_rad=angle(ar,rr),root_linear_velocity_m_s=(quat_apply_inverse(ar,asset.root_link_lin_vel_w)-quat_apply_inverse(rr,m.root_lin_vel_w[:,0])).norm(dim=-1),root_angular_velocity_rad_s=(quat_apply_inverse(ar,asset.root_link_ang_vel_w)-quat_apply_inverse(rr,m.root_ang_vel_w[:,0])).norm(dim=-1),keypoint_pos_m=(cmd._current_keypoint_pos_b-m.body_pos_b[:,0,cmd.keypoint_idx_motion]).norm(dim=-1).mean(-1),keypoint_orientation_rad=angle(cmd._current_keypoint_quat_b,quat_in_frame(rr[:,None],m.body_quat_w[:,0,cmd.keypoint_idx_motion])).mean(-1),action_rate_l2=((action-prev)**2).sum(-1),action_amplitude=action.abs().mean(-1),effort_l2=(asset.qfrc_actuator**2).sum(-1),latent_continuous_mse=((zs-zt)**2).mean(-1),teacher_student_action_rmse=((ta-sa)**2).mean(-1).sqrt(),quantization_action_rmse=((unquant-sa)**2).mean(-1).sqrt())
            ab=quat_apply_inverse(ar[:,None],asset.body_link_pos_w[:,body_a]-asset.root_link_pos_w[:,None])
            rb=quat_apply_inverse(rr[:,None],m.body_pos_w[:,0,body_m]-refroot[:,None])
            aqb=quat_in_frame(ar[:,None],asset.body_link_quat_w[:,body_a]);rqb=quat_in_frame(rr[:,None],m.body_quat_w[:,0,body_m])
            lim=asset.joint_pos_limits[:,cmd.joint_idx_asset]
            force=asset.qfrc_actuator
            force_lim=base.action_manager.model.actuator_forcerange[:,base.action_manager.torque_limit_ctrl_ids].abs().amax(-1)
            contact=contact_cache.current_for(feet_ids)
            metrics.update(body14_root_relative_pos_mm=(ab-rb).norm(dim=-1).mean(-1)*1000,body14_root_relative_ori_rad=angle(aqb,rqb).mean(-1),joint_limit_violation_fraction=((actual<lim[...,0])|(actual>lim[...,1])).float().mean(-1),effort_saturation_fraction=(force.abs()>=.99*force_lim).float().mean(-1),contact_mismatch_fraction=(contact!=cmd.feet_standing).float().mean(-1),abnormal_root_height=((root[:,2]-refroot[:,2]).abs()>.35).float(),abnormal_root_orientation=(angle(ar,rr)>1.).float())
            prev=action
            for k,v in metrics.items():
                if not torch.isfinite(v[active]).all():raise FloatingPointError(k)
                if k not in totals:totals[k]=torch.zeros(n,device=device)
                totals[k]+=torch.where(active,v,0.)
            for k in active.nonzero().flatten().tolist():
                case[k]['frames']+=1;qtrace[k].append(actual[k].cpu().numpy());rtrace[k].append(ref[k].cpu().numpy());atrace[k].append(action[k].cpu().numpy());roottrace[k].append(root[k].cpu().numpy());refroottrace[k].append(refroot[k].cpu().numpy())
            terminated=transition['next','terminated'].flatten();done=transition['next','done'].flatten()
            finishing=active & done
            for k in finishing.nonzero().flatten().tolist():
                reasons=[name for name in base.termination_funcs if bool(transition['next','stats','termination',name][k].item())]
                case[k]['termination_reasons']=reasons;case[k]['success']=completion_result(case[k]['frames'],int(lengths[k]-1),bool(terminated[k]),reasons)[0]
            active &= ~done
            if not active.any():break
            carry=env._step_mdp(transition);carry=env.maybe_reset(carry)
    for k,row in enumerate(case):
        row['completion']=min(row['frames']/row['planned_transitions'],1.)
        row['metrics']={name:float(v[k].item()/max(row['frames'],1)) for name,v in totals.items()}
        if row['frames']:
            aq=np.array(qtrace[k]);rq=np.array(rtrace[k]);aa=np.array(atrace[k]);rp=np.array(roottrace[k]);rrp=np.array(refroottrace[k]);arange=np.ptp(aq,axis=0);rrange=np.ptp(rq,axis=0)
            speed=np.linalg.norm(np.diff(rq,axis=0),axis=1)/.02
            high=speed>=np.quantile(speed,.75) if len(speed) else np.array([],dtype=bool)
            lag_errors={}
            for lag in range(-10,11):
                if len(aq)<=abs(lag)+2:continue
                left=aq[max(lag,0):len(aq)+min(lag,0)];right=rq[max(-lag,0):len(rq)-max(lag,0)]
                lag_errors[lag]=float(np.mean((left-right)**2))
            row['high_motion_joint_rmse_rad']=float(np.mean(np.sqrt(np.mean((aq[1:][high]-rq[1:][high])**2,axis=1)))) if high.any() else None
            row['best_joint_alignment_lag_ticks']=min(lag_errors,key=lag_errors.get) if lag_errors else None
            row['preservation']=dict(actual_joint_range_rad=arange.tolist(),clean_joint_range_rad=rrange.tolist(),range_ratio=float(arange.mean()/max(rrange.mean(),1e-8)),action_range=float(np.ptp(aa,axis=0).mean()),actual_root_displacement_m=float(np.linalg.norm(rp[-1]-rp[0])),clean_root_displacement_m=float(np.linalg.norm(rrp[-1]-rrp[0])),clean_high_motion_speed=float(np.quantile(abs(np.diff(rq,axis=0)),.95)/.02) if len(rq)>1 else 0.)
    summary=dict(status='PASS',scope='validation_pilot' if a.limit else 'full_split',checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),split=a.split,severity=a.severity,seed=a.seed,role=a.role,cases=len(case),success_rate=sum(r['success'] for r in case)/n,completion=float(np.mean([r['completion'] for r in case])),student_no_clean_leak=no_leak,quantized_histogram=quant_hist.cpu().tolist(),legacy_clean=a.legacy_clean,body_names=body_names,recipe=a.recipe,physics='fixed nominal plane; explicit resolved config',manifest_sha256=sha(a.data/f'{a.split}_manifest.json'),protocol_sha256=sha(a.data/'corruption_protocol.json'),metrics={name:float(np.mean([r['metrics'][name] for r in case])) for name in totals},limitations=['Errors are all-case pre-termination means; compare common-success separately.','Single initial seed; no confidence interval claimed.'])
    dump(a.output/'cases.json',case);dump(a.output/'summary.json',summary)
    np.savez_compressed(a.output/'traces.npz',**{f'{name}_{k}':np.array(tr[k]) for name,tr in [('q',qtrace),('ref',rtrace),('action',atrace),('root',roottrace),('refroot',refroottrace)] for k in range(n)})
    env.close();print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
