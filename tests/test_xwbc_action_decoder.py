import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_action_decoder import ActionDecoder
from active_adaptation.learning.modules.xwbc_staged import configure_student, student_mean, frozen_digest, Reservoir
from active_adaptation.learning.xwbc_staged import PROTOCOL, load_model, supervised_update
from scripts.refine_xwbc_student import configure, update

DIMS=dict(human=5,g1=6,roban=7)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(9)

    def model(self, kind='flow'):
        return XWBCPolicy(DIMS,dict(g1=8,roban=9),dict(g1=3,roban=2),token_dim=4,hidden=[8],encoder_hidden=[8],
            student_fsq=dict(enabled=True,scale=4.499549865722656),
            action_decoder=dict(kind=kind,width=16,steps=4,action_embed=4,time_embed=4,robot_embed=4))

    def replay(self):
        replays={}
        for r,p,a in [('g1',8,3),('roban',9,2)]:
            replays[r]=Reservoir(16)
            replays[r].add(dict(proprio=torch.randn(16,p),motion=torch.zeros(16,dtype=torch.long),
                frame=torch.zeros(16,dtype=torch.long),target=torch.ones(16,a)))
        return replays

    class Bank:
        device='cpu'
        def gather(self,motion,frame):return {s:torch.ones(len(motion),d) for s,d in DIMS.items()}

    def test_linear_conditional_path_target_and_teacher_detached(self):
        d=self.model().heads;h=torch.randn(4,8,requires_grad=True)
        target=torch.randn(4,3,requires_grad=True);noise=torch.randn(4,3);tau=torch.tensor([0.,.2,.7,.99])
        def oracle(robot,x,t,condition):
            torch.testing.assert_close(x,(1-t[:,None])*noise+t[:,None]*target.detach())
            return target.detach()-noise+condition[:,:3]*0
        with patch.object(d,'velocity',side_effect=oracle):
            loss=d.matching_loss('g1',h,target,noise=noise,tau=tau)
            self.assertEqual(float(loss),0.);loss.backward();self.assertIsNone(target.grad)

    def test_zero_euler_endpoint_for_known_field_all_native_dimensions(self):
        d=self.model().heads;h=torch.randn(5,8)
        for robot,dim in [('g1',3),('roban',2)]:
            target=torch.randn(5,dim)
            def velocity(r,x,t,condition):return (target-x)/(1-t[:,None])
            with patch.object(d,'velocity',side_effect=velocity):
                for steps in [1,2,4,8]:
                    torch.testing.assert_close(d.predict(robot,h,steps=steps),target)

    def test_private_rng_does_not_perturb_global_and_restores_exactly(self):
        d=self.model().heads;h=torch.randn(4,8);target=torch.randn(4,3)
        global_state=torch.get_rng_state().clone()
        d.matching_loss('g1',h,target)
        self.assertTrue(torch.equal(global_state,torch.get_rng_state()))
        saved=d.rng_state();expected=d.matching_loss('g1',h,target)
        d.restore_rng(saved);actual=d.matching_loss('g1',h,target)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        before=d.rng_state();d.predict('g1',h)
        self.assertTrue(all(torch.equal(v,d.rng_state()[k]) for k,v in before.items()))

    def test_common_weights_and_global_initialization_rng_identical(self):
        torch.manual_seed(4);a=self.model('flow');ra=torch.get_rng_state()
        torch.manual_seed(4);b=self.model('mlp');rb=torch.get_rng_state()
        self.assertTrue(torch.equal(ra,rb))
        for k,v in a.state_dict().items():
            if not k.startswith('heads.'):self.assertTrue(torch.equal(v,b.state_dict()[k]),k)

    def test_checkpoint_roundtrip_and_no_fake_gaussian_flow(self):
        for kind in ('flow','mlp'):
            m=self.model(kind);x=torch.randn(4,8);c=torch.randn(4,5)
            for stage in ('B','E'):
                with tempfile.TemporaryDirectory() as directory:
                    p=Path(directory)/'m.pt'
                    torch.save(dict(format=PROTOCOL,stage=stage,constructor=m.constructor,model=m.state_dict()),p)
                    loaded,_=load_model(p,'cpu')
                    torch.testing.assert_close(student_mean(m,'g1',x,c,'human'),student_mean(loaded,'g1',x,c,'human'),rtol=0,atol=0)
            if kind=='flow':
                with self.assertRaisesRegex(RuntimeError,'Gaussian'):
                    m.actor('g1',x,torch.zeros(4,dtype=torch.long),{s:torch.randn(4,d) for s,d in DIMS.items()})

    def test_B_flow_supervision_reaches_encoder_and_freezes_others(self):
        m=self.model();replay=self.replay();params=configure_student(m,'B');frozen=frozen_digest(m)
        before={k:v.clone() for k,v in m.state_dict().items()}
        cfg=dict(supervised_steps=5,batch_size=8,max_grad_norm=1)
        metrics=supervised_update('B',m,torch.optim.Adam(params,lr=.003),self.Bank(),replay,cfg)
        self.assertIn('g1/flow_mse',metrics);self.assertEqual(frozen_digest(m),frozen)
        for prefix in ['encoders.shared.human.','heads.action_inputs.g1.','heads.action_inputs.roban.','heads.body.']:
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith(prefix)),prefix)

    def test_E_updates_all_sources_and_preserves_anchor_for_both_decoders(self):
        for kind in ('flow','mlp'):
            m=self.model(kind);anchor=copy.deepcopy(m).requires_grad_(False)
            before={k:v.clone() for k,v in m.state_dict().items()}
            opt=torch.optim.Adam(configure(m,True),lr=.003);frozen=frozen_digest(m)
            metrics=update(m,anchor,opt,self.Bank(),self.replay(),dict(supervised_steps=5,batch_size=8,anchor_weight=.1,max_grad_norm=1))
            self.assertEqual(frozen_digest(m),frozen)
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in anchor.state_dict().items()))
            for source in DIMS:
                self.assertTrue(any(not torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith('encoders.shared.'+source+'.')))
            if kind=='flow':self.assertIn('roban/human/flow_mse',metrics)

    def test_anchor_backpropagates_through_full_solver(self):
        m=self.model();configure(m,True)
        with patch.object(m.heads,'matching_loss',side_effect=lambda r,h,t:h.sum()*0):
            anchor=copy.deepcopy(m).requires_grad_(False)
            with torch.no_grad():
                for head in anchor.heads.outputs.values():head.bias.add_(1)
            before=m.heads.outputs['g1'].weight.detach().clone()
            opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
            update(m,anchor,opt,self.Bank(),self.replay(),dict(supervised_steps=1,batch_size=8,anchor_weight=.1,max_grad_norm=1))
            self.assertFalse(torch.equal(before,m.heads.outputs['g1'].weight))

    def test_actual_capacity_matched_with_native_29_and_21_dimensions(self):
        dims=dict(g1=29,roban=21)
        a=ActionDecoder(256,dims,'flow',width=256)
        b=ActionDecoder(256,dims,'mlp',width=288)
        ca=sum(p.numel() for p in a.parameters());cb=sum(p.numel() for p in b.parameters())
        self.assertLess(abs(ca-cb)/ca,.01)


if __name__=='__main__':unittest.main()
