import copy
import unittest
import torch
from scripts.refine_xwbc_student import iteration_config, configure, update
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import Reservoir, student_mean


class NoCAblationTest(unittest.TestCase):
    def test_actual_config_resolves_dataset_and_teacher_settings(self):
        from active_adaptation.learning.xwbc import load_train_config
        from pathlib import Path
        cfg=load_train_config(Path(__file__).resolve().parents[1]/'cfg/xwbc/v02/teacher_refinement_no_c.yaml')
        self.assertEqual(cfg['parent_stage'],'B')
        self.assertTrue(cfg['train_sources'])
        self.assertEqual(cfg['learning_rate'],1e-5)
        self.assertIn('dataset',cfg)
        self.assertEqual(iteration_config(cfg,2001)['batch_size'],1024)

    def test_budget_matches_c_plus_e_and_switches_after_2000(self):
        cfg = dict(updates=2300, supervised_steps=16, batch_size=1024,
                   budget_schedule=[dict(until_update=2000, supervised_steps=32, batch_size=2048),
                                    dict(until_update=2300, supervised_steps=16, batch_size=1024)])
        settings = [iteration_config(cfg, i) for i in range(1,2301)]
        self.assertEqual(sum(x['supervised_steps'] for x in settings), 68800)
        self.assertEqual(sum(x['supervised_steps']*x['batch_size']*2 for x in settings),271974400)
        self.assertEqual(settings[1999]['batch_size'],2048)
        self.assertEqual(settings[2000]['batch_size'],1024)
        with self.assertRaises(ValueError):
            iteration_config(dict(cfg,updates=2301),1)

    def test_random_sources_learn_teacher_with_fixed_initial_student_anchor(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        dims=dict(human=5,g1=6,roban=7)
        model=XWBCPolicy(dims,dict(g1=8,roban=9),dict(g1=3,roban=2),token_dim=4,hidden=[8],encoder_hidden=[8])
        anchor=copy.deepcopy(model).requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.Adam(configure(model,True),lr=.003)
        class Bank:
            device='cpu'
            def gather(self,motion,frame):
                return {s:torch.ones(len(motion),n) for s,n in dims.items()}
        replay={}
        for r,n in [('g1',8),('roban',9)]:
            replay[r]=Reservoir(16)
            replay[r].add(dict(proprio=torch.ones(16,n),motion=torch.zeros(16,dtype=torch.long),
                               frame=torch.zeros(16,dtype=torch.long),target=torch.ones(16,3 if r=='g1' else 2)))
        update(model,anchor,optimizer,Bank(),replay,dict(supervised_steps=30,batch_size=8,anchor_weight=.1,max_grad_norm=1.))
        for k,v in anchor.state_dict().items():
            self.assertTrue(torch.equal(v,before[k]))
        for name,value in model.state_dict().items():
            if name.startswith(('critics.','log_std.')):
                self.assertTrue(torch.equal(value,before[name]))
        for r,n in [('g1',8),('roban',9)]:
            for source,d in dims.items():
                args=(r,torch.ones(2,n),torch.ones(2,d),source)
                self.assertLess(float((student_mean(model,*args)-1).square().mean().detach()),
                                float((student_mean(anchor,*args)-1).square().mean()))
        for source in dims:
            prefix='encoders.shared.'+source+'.'
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(prefix)))
