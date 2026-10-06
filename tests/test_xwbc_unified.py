import unittest
import torch
from scripts.train_xwbc_unified import assignments,source_probabilities,takeover,learning_rate
from scripts.refine_xwbc_student import iteration_config,configure,update
from active_adaptation.learning.xwbc import load_train_config
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import Reservoir
from pathlib import Path

class UnifiedTests(unittest.TestCase):
 def test_schedules_and_budget(self):
  root=Path(__file__).resolve().parents[1]
  for variant in ('uniform','curriculum'):
   c=load_train_config(root/f'cfg/xwbc/v02/unified_{variant}.yaml')
   self.assertEqual(sum(iteration_config(c,i)['supervised_steps'] for i in range(1,4301)),132800)
   self.assertEqual(sum(iteration_config(c,i)['supervised_steps']*iteration_config(c,i)['batch_size']*2 for i in range(1,4301)),534118400)
   self.assertEqual(takeover(c,1),1);self.assertEqual(takeover(c,521),0)
   self.assertAlmostEqual(learning_rate(c,1),1e-4);self.assertAlmostEqual(learning_rate(c,4300),1e-5)
   self.assertAlmostEqual(source_probabilities(c,1500)[0],1/3)
   self.assertAlmostEqual(source_probabilities(c,1)[0],.6 if variant=='curriculum' else 1/3)
 def test_assignments_preserve_active_and_fill_deficits(self):
  a=assignments([0,0,0],[.6,.2,.2],100,100);self.assertEqual([a.count(i) for i in range(3)],[60,20,20])
  b=assignments([60,10,10],[1/3]*3,100,20);self.assertNotIn(0,b)
  self.assertEqual([60+b.count(0),10+b.count(1),10+b.count(2)],[60,20,20])
 def test_pure_teacher_no_anchor_updates_all_sources(self):
  torch.set_num_threads(1);torch.manual_seed(4)
  dims=dict(human=5,g1=6,roban=7);m=XWBCPolicy(dims,dict(g1=8,roban=9),dict(g1=3,roban=2),token_dim=4,hidden=[8],encoder_hidden=[8])
  before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.Adam(configure(m,True),lr=.003)
  class Bank:
   device='cpu'
   def gather(self,motion,frame):return {s:torch.ones(len(motion),d) for s,d in dims.items()}
  replays={}
  for r,n in [('g1',8),('roban',9)]:
   replays[r]=Reservoir(16);replays[r].add(dict(proprio=torch.ones(16,n),motion=torch.zeros(16,dtype=torch.long),frame=torch.zeros(16,dtype=torch.long),target=torch.ones(16,3 if r=='g1' else 2)))
  result=update(m,None,opt,Bank(),replays,dict(supervised_steps=10,batch_size=8,anchor_weight=0,max_grad_norm=1))
  for s in dims:self.assertTrue(any(not torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith('encoders.shared.'+s+'.')))
  for k,v in m.state_dict().items():
   if k.startswith(('critics.','log_std.')):self.assertTrue(torch.equal(v,before[k]))
  self.assertTrue(all(v==0 for k,v in result.items() if k.endswith('anchor_mse')))
