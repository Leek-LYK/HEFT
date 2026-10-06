import copy,tempfile,unittest
from pathlib import Path
import torch
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import NativeTeacher,configure_student,student_mean,frozen_digest
from active_adaptation.learning.xwbc_staged import load_model,PROTOCOL,gate_decision,stage_name
from scripts.refine_xwbc_student import configure,update
from active_adaptation.learning.modules.xwbc_staged import Reservoir

class TransformerTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(5);self.cfg=dict(d_model=16,nhead=4,layers=2,ff_dim=32)
 def model(self):return XWBCPolicy(dict(human=5,g1=6,roban=7),dict(g1=8,roban=9),dict(g1=3,roban=2),token_dim=4,hidden=[8],encoder_hidden=[8],transformer=self.cfg)
 def test_teacher_attention_gradient_and_checkpoint_roundtrip(self):
  m=NativeTeacher('g1',6,8,4,3,hidden=[16],transformer=self.cfg);x,p,c=torch.randn(4,8),torch.randn(4,4),torch.randn(4,6)
  loss=m.distribution(x,p,c).mean.square().sum()+m.value(x,p,c).square().sum();loss.backward()
  self.assertGreater(float(m.actor_body.blocks[0].qkv.weight.grad.abs().sum()),0)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'t.pt';torch.save(dict(format=PROTOCOL,stage='A_T',constructor=m.constructor,model=m.state_dict()),path)
   loaded,_=load_model(path,'cpu');torch.testing.assert_close(m.distribution(x,p,c).mean,loaded.distribution(x,p,c).mean)
 def test_student_train_eval_and_roundtrip(self):
  m=self.model();x=torch.randn(4,8);c=torch.randn(4,5)
  m.train();a=student_mean(m,'g1',x,c,'human');m.eval();torch.testing.assert_close(a,student_mean(m,'g1',x,c,'human'))
  for stage in ['B_T','E_T']:
   with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'s.pt';torch.save(dict(format=PROTOCOL,stage=stage,constructor=m.constructor,model=m.state_dict()),path)
    loaded,_=load_model(path,'cpu');torch.testing.assert_close(a,student_mean(loaded,'g1',x,c,'human'))
 def test_B_freezes_robot_sources_E_trains_all_and_keeps_anchor(self):
  m=self.model();configure_student(m,'B');frozen=frozen_digest(m)
  opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
  opt.zero_grad();student_mean(m,'g1',torch.ones(4,8),torch.ones(4,5),'human').sum().backward();opt.step()
  self.assertEqual(frozen_digest(m),frozen)
  anchor=copy.deepcopy(m).requires_grad_(False);before={k:v.clone() for k,v in anchor.state_dict().items()}
  opt=torch.optim.Adam(configure(m,True),lr=.003);replays={}
  class Bank:
   device='cpu'
   def gather(self,motion,frame):return {s:torch.ones(len(motion),d) for s,d in dict(human=5,g1=6,roban=7).items()}
  for r,n in [('g1',8),('roban',9)]:
   replays[r]=Reservoir(16);replays[r].add(dict(proprio=torch.ones(16,n),motion=torch.zeros(16,dtype=torch.long),frame=torch.zeros(16,dtype=torch.long),target=torch.ones(16,3 if r=='g1' else 2)))
  frozen=frozen_digest(m);update(m,anchor,opt,Bank(),replays,dict(supervised_steps=20,batch_size=8,anchor_weight=.1,max_grad_norm=1))
  self.assertEqual(frozen_digest(m),frozen)
  self.assertTrue(all(torch.equal(v,before[k]) for k,v in anchor.state_dict().items()))
  for source in ['human','g1','roban']:self.assertTrue(any(not torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith('encoders.shared.'+source+'.')))
 def test_stage_names_and_gate_alias(self):
  self.assertEqual(stage_name('A',dict(transformer=self.cfg)),'A_T');self.assertEqual(stage_name('B',{}),'B')
  # Synthetic complete route: same scoring decision for A and A_T.
  a=dict(teacher_success=.9,teacher_completion=.95,max_recovery_clip_fraction=.01,teacher_relative_mpkpe=.1,teacher_world_mpkpe=1.)
  s=dict(stage='A',robot='g1',split='val',full_scope=True,expected_clips=500,acceptance=a,matrix={'g1->g1':dict(clips=500,success_rate=1.,completion=1.,recovery_clips=0,mpkpe_root_relative_m=.05,mpkpe_world_m=.5)})
  expected=gate_decision(s);s['stage']='A_T';self.assertEqual(gate_decision(s),expected)
