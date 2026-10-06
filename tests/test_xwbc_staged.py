import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import (
    NativeTeacher,Reservoir,configure_student,action_mse,stage_c_loss,frozen_digest,student_mean)
from active_adaptation.learning.xwbc_staged import (
    PROTOCOL,dagger_collect,gate_decision,load_model,takeover_probability,checkpoint,restore_optimizer,verify_gate,identity_hash)
from active_adaptation.utils.xwbc_data import sha256

class StagedTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(7)

    def student(self):
        return XWBCPolicy(dict(human=5,g1=6,roban=7),dict(g1=8,roban=9),dict(g1=3,roban=2),token_dim=4,hidden=[8],encoder_hidden=[8])

    def commands(self,n=4):return {s:torch.randn(n,d) for s,d in dict(human=5,g1=6,roban=7).items()}

    def test_teacher_native_privileged_model_roundtrip(self):
        t=NativeTeacher('g1',6,8,4,3,hidden=[16])
        x,p,c=torch.randn(4,8),torch.randn(4,4),torch.randn(4,6)
        self.assertEqual(t.distribution(x,p,c).mean.shape,(4,3))
        self.assertFalse(hasattr(t,'encoders'))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'teacher.pt'
            torch.save(dict(format=PROTOCOL,stage='A',constructor=t.constructor,model=t.state_dict()),path)
            loaded,_=load_model(path,'cpu')
            torch.testing.assert_close(t.distribution(x,p,c).mean,loaded.distribution(x,p,c).mean,rtol=0,atol=0)

    def test_stage_b_only_human_and_execution_parameters_train(self):
        m=self.student();configure_student(m,'B')
        names={k for k,v in m.named_parameters() if v.requires_grad}
        self.assertTrue(names)
        self.assertTrue(all(k.startswith(('encoders.shared.human.','backbones.shared.','proprio.','heads.')) for k in names))
        mean=student_mean(m,'g1',torch.randn(4,8),torch.randn(4,5),'human')
        action_mse(mean,torch.ones_like(mean)).backward()
        self.assertTrue(all(p.grad is None for p in m.encoders['shared']['g1'].parameters()))
        self.assertTrue(all(p.grad is None for p in m.critics.parameters()))

    def test_actions_are_dimension_normalized_and_teacher_detached(self):
        teacher_g=torch.ones(4,29,requires_grad=True);teacher_r=torch.ones(4,21,requires_grad=True)*2
        g=torch.zeros(4,29,requires_grad=True);r=torch.zeros(4,21,requires_grad=True)
        loss=(action_mse(g,teacher_g)+action_mse(r,teacher_r))/2
        self.assertAlmostEqual(float(loss),2.5)
        loss.backward();self.assertIsNone(teacher_g.grad)
        self.assertAlmostEqual(float(g.grad.sum()),-1.,places=6)
        self.assertAlmostEqual(float(r.grad.sum()),-2.,places=6)

    def test_stage_c_frozen_executor_and_human_path_unchanged(self):
        m=self.student();params=configure_student(m,'C');before=frozen_digest(m)
        self.assertTrue(all(k.startswith(('encoders.shared.g1.','encoders.shared.roban.')) for k,p in m.named_parameters() if p.requires_grad))
        commands=self.commands();obs=torch.randn(4,8)
        with torch.no_grad():anchor=student_mean(m,'g1',obs,commands['human'],'human').clone()
        encoder_before={k:p.clone() for k,p in m.encoders['shared']['g1'].state_dict().items()}
        opt=torch.optim.Adam(params,lr=.01)
        for _ in range(3):
            opt.zero_grad();loss,_,_=stage_c_loss(m,'g1',obs,commands,anchor,1.,1.);loss.backward();opt.step()
        self.assertEqual(frozen_digest(m),before)
        torch.testing.assert_close(student_mean(m,'g1',obs,commands['human'],'human'),anchor,rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,encoder_before[k]) for k,v in m.encoders['shared']['g1'].state_dict().items()))

    def test_stage_c_matches_raw_token_amplitude(self):
        class Scale(nn.Module):
            def __init__(self,factor):super().__init__();self.factor=nn.Parameter(torch.tensor(float(factor)))
            def forward(self,x):return self.factor*x
        class Zeros(nn.Module):
            def forward(self,x):return torch.zeros(len(x),2)
        head=nn.Linear(4,1,bias=False)
        with torch.no_grad():head.weight.copy_(torch.tensor([[1.,1.,0.,0.]]))
        head.requires_grad_(False)
        m=SimpleNamespace(encoders={'shared':{'human':Scale(1),'g1':Scale(2),'roban':Scale(1)}},
                          proprio={'g1':Zeros()},backbones={'shared':nn.Identity()},heads={'g1':head})
        command={s:torch.ones(4,2) for s in ['human','g1','roban']}
        _,token,act=stage_c_loss(m,'g1',torch.zeros(4,2),command,torch.full((4,1),2.),1.,1.)
        self.assertAlmostEqual(float(token),1.)
        self.assertAlmostEqual(float(act),4.)

    def test_reservoir_aggregation_and_resume(self):
        buf=Reservoir(7,3)
        buf.add(dict(state=torch.arange(100)[:,None].float(),target=torch.arange(100)[:,None].float()*2))
        self.assertEqual(buf.seen,100);self.assertEqual(buf.size,7)
        self.assertTrue((buf.data['state']<90).any())
        restored=Reservoir(7);restored.load_state_dict(buf.state_dict())
        for b in [buf,restored]:b.add(dict(state=torch.arange(100,140)[:,None].float(),target=torch.arange(100,140)[:,None].float()*2))
        for k,v in buf.sample(20,'cpu').items():
            # Compare a full sample below to preserve the same RNG draw per reservoir.
            self.assertEqual(len(v),20)
        restored.generator.set_state(buf.generator.get_state())
        a=buf.sample(20,'cpu');b=restored.sample(20,'cpu')
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        torch.testing.assert_close(a['target'],a['state']*2)

    def test_dagger_queries_same_student_visited_state_and_executes_one_history(self):
        class Pool:
            robot_name='g1';num_envs=2
            def __init__(self):self.state=torch.zeros(2,1);self.previous=torch.zeros(2,1);self.source=torch.zeros(2,dtype=torch.long);self.executed=[]
            def observe(self):return dict(proprio=torch.cat((self.state,self.previous),1),motion=torch.zeros(2,dtype=torch.long),frame=torch.zeros(2,dtype=torch.long),source=self.source)
            def teacher_observation(self,obs):return obs['proprio'][:,:1]*10
            def step(self,action,mean):
                torch.testing.assert_close(action,mean)
                self.executed.append(action.clone());self.state+=action;self.previous=action.clone()
                return {'reward':torch.ones(2,1),'done':torch.zeros(2,1,dtype=torch.bool)},self.observe()
            def reset(self,ids):self.assert_empty=ids.numel()==0
        class Teacher:
            robot='g1'
            def __init__(self):self.queries=[]
            def distribution(self,proprio,privileged,command):
                self.queries.append(proprio.clone())
                return SimpleNamespace(mean=proprio[:,:1]+command[:,:1])
        bank=SimpleNamespace(device='cpu',gather=lambda motion,frame:{'human':torch.ones(2,1),'g1':torch.full((2,1),100.),'roban':torch.full((2,1),-100.)})
        pool=Pool();teacher=Teacher();replay=Reservoir(20)
        with patch('active_adaptation.learning.xwbc_staged.student_mean',return_value=torch.ones(2,1)):
            stats=dagger_collect('B',None,{'g1':teacher},pool,bank,replay,3,beta=0)
        self.assertEqual(stats['takeover_fraction'],0)
        self.assertEqual([float(x[0,0]) for x in teacher.queries],[0,1,2])
        self.assertEqual([float(x[0,1]) for x in teacher.queries],[0,1,1])
        torch.testing.assert_close(replay.data['target'][:6,0],torch.tensor([100.,100.,101.,101.,102.,102.]))
        self.assertTrue(all(torch.equal(a,torch.ones(2,1)) for a in pool.executed))

    def gate_summary(self):
        metrics=dict(clips=500,success_rate=.95,completion=.98,recovery_clips=0,mpkpe_root_relative_m=.04,mpkpe_world_m=.05)
        acceptance=dict(teacher_success=.90,teacher_completion=.95,max_recovery_clip_fraction=.01,teacher_relative_mpkpe=.1,
                        student_success_drop=.05,student_completion_drop=.05,student_relative_error_increase_m=.02)
        return dict(stage='A',robot='g1',split='val',full_scope=True,expected_clips=500,matrix={'g1->g1':metrics},acceptance=acceptance)

    def test_gate_requires_complete_validation_and_real_quality(self):
        s=self.gate_summary();self.assertTrue(gate_decision(s)['passed'])
        for change in ['partial','low_success','nan','test_split','standing_with_world_drift']:
            altered=copy.deepcopy(s)
            if change=='partial':altered['full_scope']=False
            elif change=='low_success':altered['matrix']['g1->g1']['success_rate']=.2
            elif change=='nan':altered['matrix']['g1->g1']['completion']=float('nan')
            elif change=='standing_with_world_drift':altered['matrix']['g1->g1']['mpkpe_world_m']=2.
            else:altered['split']='test'
            self.assertFalse(gate_decision(altered)['passed'])

    def test_world_diagnostic_retains_quality_and_integrity_requirements(self):
        s=self.gate_summary();s['acceptance']['world_error_is_diagnostic']=True
        s['matrix']['g1->g1']['mpkpe_world_m']=2.
        decision=gate_decision(s)
        self.assertTrue(decision['passed'])
        self.assertEqual(decision['diagnostics'],['g1->g1:world_tracking_error'])
        for field,value in [('success_rate',.2),('completion',.2),('recovery_clips',20),
                            ('mpkpe_root_relative_m',.2),('mpkpe_world_m',float('nan'))]:
            altered=copy.deepcopy(s);altered['matrix']['g1->g1'][field]=value
            self.assertFalse(gate_decision(altered)['passed'],field)
        s['full_scope']=False
        self.assertFalse(gate_decision(s)['passed'])

    def test_student_world_diagnostic_keeps_success_drop_gate(self):
        s=self.gate_summary();m=copy.deepcopy(s['matrix']['g1->g1'])
        s['acceptance']['world_error_is_diagnostic']=True
        for stage,sources in [('B',['human']),('C',['human','g1','roban'])]:
            s.update(stage=stage,robot=None,baselines={r:copy.deepcopy(m) for r in ['g1','roban']},
                     matrix={a+'->'+r:copy.deepcopy(m) for a in sources for r in ['g1','roban']})
            s['matrix']['human->roban']['mpkpe_world_m']=2.
            decision=gate_decision(s)
            self.assertTrue(decision['passed'])
            self.assertIn('human->roban:world_tracking_error_increase',decision['diagnostics'])
            s['matrix']['human->roban']['success_rate']=.7
            self.assertIn('human->roban:success_drop',gate_decision(s)['issues'])

    def test_stage_b_capability_gate_preserves_original_diagnostics(self):
        s=self.gate_summary();m=copy.deepcopy(s['matrix']['g1->g1'])
        m.update(success_rate=.994,completion=.997,mpkpe_root_relative_m=.044)
        s.update(stage='B',robot=None,baselines={r:copy.deepcopy(m) for r in ['g1','roban']},
                 matrix={'human->'+r:copy.deepcopy(m) for r in ['g1','roban']})
        for v in s['matrix'].values():v.update(success_rate=.91,completion=.96,mpkpe_root_relative_m=.09,mpkpe_world_m=.9)
        s['acceptance']['world_error_is_diagnostic']=True
        self.assertFalse(gate_decision(s)['passed'])
        s['acceptance']['stage_b_absolute']=dict(success=.90,completion=.95,relative_mpkpe=.10)
        decision=gate_decision(s)
        self.assertTrue(decision['passed'])
        self.assertIn('human->g1:success_drop',decision['diagnostics'])
        self.assertIn('human->g1:tracking_error_increase',decision['diagnostics'])
        for field,value in [('success_rate',.899),('completion',.949),('recovery_clips',6),
                            ('mpkpe_root_relative_m',.101),('mpkpe_world_m',float('nan'))]:
            altered=copy.deepcopy(s);altered['matrix']['human->g1'][field]=value
            self.assertFalse(gate_decision(altered)['passed'],field)
        altered=copy.deepcopy(s);altered['full_scope']=False
        self.assertFalse(gate_decision(altered)['passed'])
        altered=copy.deepcopy(s);del altered['matrix']['human->roban']
        self.assertIn('route_coverage',gate_decision(altered)['issues'])

    def test_stage_b_capability_setting_does_not_relax_stage_c(self):
        s=self.gate_summary();m=copy.deepcopy(s['matrix']['g1->g1'])
        m.update(success_rate=.99,completion=.99,mpkpe_root_relative_m=.04)
        s['acceptance']['stage_b_absolute']=dict(success=.90,completion=.95,relative_mpkpe=.10)
        s.update(stage='C',robot=None,baselines={r:copy.deepcopy(m) for r in ['g1','roban']},
                 matrix={a+'->'+r:copy.deepcopy(m) for a in ['human','g1','roban'] for r in ['g1','roban']})
        self.assertTrue(gate_decision(s)['passed'])
        s['matrix']['g1->roban'].update(success_rate=.91,mpkpe_root_relative_m=.09)
        decision=gate_decision(s)
        self.assertIn('g1->roban:success_drop',decision['issues'])
        self.assertIn('g1->roban:tracking_error_increase',decision['issues'])

    def test_takeover_schedule_reaches_student_control(self):
        cfg=dict(stage_b=dict(warmup_updates=20,teacher_decay_updates=500),stage_c=dict(human_decay_updates=300))
        self.assertEqual(takeover_probability('B',0,cfg),1)
        self.assertEqual(takeover_probability('B',520,cfg),0)
        self.assertEqual(takeover_probability('C',300,cfg),0)

    def test_b_and_c_gates_use_target_baselines_and_all_routes(self):
        s=self.gate_summary();m=copy.deepcopy(s['matrix']['g1->g1'])
        s.update(stage='B',robot=None,baselines={r:copy.deepcopy(m) for r in ['g1','roban']},
                 matrix={'human->'+r:copy.deepcopy(m) for r in ['g1','roban']})
        self.assertTrue(gate_decision(s)['passed'])
        s['matrix']['human->roban']['success_rate']=.7
        self.assertFalse(gate_decision(s)['passed'])
        s.update(stage='C',matrix={a+'->'+r:copy.deepcopy(m) for a in ['human','g1','roban'] for r in ['g1','roban']})
        self.assertTrue(gate_decision(s)['passed'])
        del s['matrix']['g1->roban']
        self.assertIn('route_coverage',gate_decision(s)['issues'])

    def test_gate_is_bound_to_checkpoint_data_and_motion_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'val/shard').mkdir(parents=True)
            rows=[{'motion_id':f'm{i}'} for i in range(500)]
            (root/'val/shard/motions.json').write_text(json.dumps(rows))
            (root/'dataset.json').write_text(json.dumps({'splits':{'val':[{'path':'val/shard'}]}}))
            ckpt=root/'teacher.pt';ckpt.write_bytes(b'synthetic checkpoint identity')
            cfg=dict(dataset=str(root),command_offsets=[0],history_steps=[0],action_history=1)
            summary=self.gate_summary();cfg['acceptance']=summary['acceptance']
            summary.update(protocol=PROTOCOL,checkpoint_sha256=sha256(ckpt),dataset_digest=sha256(root/'dataset.json'),
                           context={k:cfg[k] for k in ['command_offsets','history_steps','action_history']},motion_ids_digest=identity_hash(rows))
            (root/'summary.json').write_text(json.dumps(summary));gate=root/'gate.json'
            gate.write_text(json.dumps({'passed':True,'summary_sha256':sha256(root/'summary.json')}))
            self.assertEqual(verify_gate(gate,ckpt,cfg,'A','g1')['stage'],'A')
            ckpt.write_bytes(b'changed teacher')
            with self.assertRaisesRegex(ValueError,'checkpoint mismatch'):verify_gate(gate,ckpt,cfg,'A','g1')

    def test_supervised_checkpoint_restores_optimizer_and_aggregated_data(self):
        cfg=dict(num_envs=2,rollout_steps=2,command_offsets=[0],history_steps=[0],action_history=1,learning_rate=.001)
        parents={r:{'sha256':r} for r in ['g1','roban']}
        m=self.student();opt=torch.optim.Adam(configure_student(m,'B'),lr=.001)
        replay={r:Reservoir(8) for r in ['g1','roban']}
        for b in replay.values():b.add({'proprio':torch.randn(4,2),'target':torch.randn(4,1)})
        loss=student_mean(m,'g1',torch.randn(4,8),torch.randn(4,5),'human').square().mean();loss.backward();opt.step()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';checkpoint(path,m,opt,'B',3,cfg,'digest',parents,{},replay)
            restored=self.student();new_opt=torch.optim.Adam(configure_student(restored,'B'),lr=.001)
            new_replay={r:Reservoir(8) for r in replay}
            step=restore_optimizer(SimpleNamespace(resume=path),restored,new_opt,cfg,'B','digest',new_replay,parents)
            self.assertEqual(step,3);self.assertTrue(new_opt.state)
            for k,v in m.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            for r in replay:
                a=replay[r].sample(5,'cpu');b=new_replay[r].sample(5,'cpu')
                for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
