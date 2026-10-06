"""Behavior checks for teacher correction without privileged student inputs."""
import copy
import unittest
import torch
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import Reservoir, frozen_digest, student_mean
from scripts.refine_xwbc_student import configure, update


class RefinementTest(unittest.TestCase):
    def test_all_source_variant_opens_only_actor_parameters(self):
        torch.set_num_threads(1)
        model=XWBCPolicy(dict(human=5,g1=6,roban=7),dict(g1=8,roban=9),dict(g1=3,roban=2),
                         token_dim=4,hidden=[8],encoder_hidden=[8])
        configure(model,True)
        for source,dimension in dict(human=5,g1=6,roban=7).items():
            mean=student_mean(model,'g1',torch.ones(2,8),torch.ones(2,dimension),source)
            mean.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.encoders['shared'][source].parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.critics.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.log_std.parameters()))

    def test_six_routes_learn_teacher_labels_without_changing_anchor_or_sources(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        dims=dict(human=5,g1=6,roban=7)
        model=XWBCPolicy(dims,dict(g1=8,roban=9),dict(g1=3,roban=2),
                         token_dim=4,hidden=[8],encoder_hidden=[8])
        anchor=copy.deepcopy(model).requires_grad_(False)
        params=configure(model)
        fixed=frozen_digest(model)
        anchor_before={k:v.clone() for k,v in anchor.state_dict().items()}
        class Bank:
            device='cpu'
            def gather(self,motion,frame):
                return {s:torch.ones(len(motion),n) for s,n in dims.items()}
        bank=Bank()
        replay={}
        for r,n in [('g1',8),('roban',9)]:
            replay[r]=Reservoir(16)
            replay[r].add(dict(proprio=torch.ones(16,n),motion=torch.zeros(16,dtype=torch.long),
                               frame=torch.zeros(16,dtype=torch.long),target=torch.ones(16,3 if r=='g1' else 2)))
        optimizer=torch.optim.Adam(params,lr=.003)
        result=update(model,anchor,optimizer,bank,replay,dict(supervised_steps=30,batch_size=8,
                      anchor_weight=.1,max_grad_norm=1.))
        self.assertEqual(frozen_digest(model),fixed)
        for k,v in anchor.state_dict().items():
            self.assertTrue(torch.equal(v,anchor_before[k]))
        for r,n in [('g1',8),('roban',9)]:
            for s,d in dims.items():
                old=student_mean(anchor,r,torch.ones(2,n),torch.ones(2,d),s)
                new=student_mean(model,r,torch.ones(2,n),torch.ones(2,d),s)
                self.assertLess(float((new-1).square().mean()),float((old-1).square().mean()))
                self.assertIn(r+'/'+s+'/teacher_mse',result)


if __name__=='__main__':
    unittest.main()
