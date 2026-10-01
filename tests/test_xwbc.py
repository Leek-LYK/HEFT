import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from active_adaptation.learning.modules.xwbc import XWBCPolicy, alignment_loss, gae, ppo_loss
from active_adaptation.utils.xwbc_data import CommandBank, heading_features, validate_manifest, sha256


class XWBCTest(unittest.TestCase):
    def make_policy(self, variant='shared'):
        return XWBCPolicy(dict(human=5,g1=6,roban=7),dict(g1=8,roban=9),dict(g1=3,roban=2),
                          token_dim=4,hidden=[8],variant=variant)

    def test_actor_routes_exclude_other_sources(self):
        p=self.make_policy()
        inputs={s:torch.randn(4,n) for s,n in dict(human=5,g1=6,roban=7).items()}
        proprio=torch.randn(4,8); route=torch.zeros(4,dtype=torch.long)
        first=p.actor('g1',proprio,route,inputs)[0].mean
        inputs['g1']*=100;inputs['roban']*=100
        torch.testing.assert_close(first,p.actor('g1',proprio,route,inputs)[0].mean,rtol=0,atol=0)
        self.assertEqual(p.actor('roban',torch.randn(4,9),route,inputs)[0].mean.shape,(4,2))

    def test_shared_and_independent_gradient_routing(self):
        for variant in ['shared','independent']:
            p=self.make_policy(variant)
            for robot,dim in [('g1',8),('roban',9)]:
                p.zero_grad(set_to_none=True)
                commands={s:torch.randn(4,n) for s,n in dict(human=5,g1=6,roban=7).items()}
                p.actor(robot,torch.randn(4,dim),torch.zeros(4,dtype=torch.long),commands)[0].mean.sum().backward()
                group=robot if variant=='independent' else 'shared'
                self.assertGreater(sum(float(x.grad.abs().sum()) for x in p.backbones[group].parameters()),0)
                other='roban' if robot=='g1' else 'g1'
                self.assertTrue(all(x.grad is None for x in p.heads[other].parameters()))

    def test_gae_timeout_bootstrap_and_reset_boundary(self):
        reward=torch.tensor([[1.,1.],[100.,100.]])
        value=torch.zeros_like(reward);nxt=torch.ones_like(reward)*2
        done=torch.tensor([[True,True],[True,True]])
        terminal=torch.tensor([[True,False],[True,True]])
        advantage,_=gae(reward,value,nxt,done,terminal,gamma=.9)
        torch.testing.assert_close(advantage[0],torch.tensor([1.,2.8]))

    def test_alignment_and_logprob_use_valid_action_dims(self):
        z=torch.randn(4,1,8).expand(-1,3,-1)
        self.assertAlmostEqual(float(alignment_loss(z)),0)
        dist=torch.distributions.Normal(torch.zeros(4,2),torch.ones(4,2))
        action=torch.zeros(4,2); old=dist.log_prob(action).sum(-1)
        loss,_,kl=ppo_loss(dist,action,old,torch.ones(4),.2)
        self.assertAlmostEqual(float(loss),-1)
        self.assertAlmostEqual(float(kl),0)

    def test_family_leakage_rejected(self):
        base=dict(family_id='walk',paths=dict(human='h',g1='g',roban='r'))
        with self.assertRaisesRegex(ValueError,'family leakage'):
            validate_manifest([dict(base,motion_id='a',split='train'),dict(base,motion_id='b',split='test')])

    def test_research_config_has_no_loads_and_covers_semantic_points(self):
        import re
        from active_adaptation.envs.xwbc import environment_config
        from active_adaptation.utils.xwbc_data import BODIES
        class Bank:
            lengths = torch.tensor([20])
            def groups(self, robot):
                return [dict(name='test',teacher_mem_path='/tmp/example',weight=1.)]
        for robot in ['g1','roban']:
            cfg=environment_config(robot,Bank(),4)
            self.assertEqual(dict(cfg.randomization),{})
            self.assertEqual(list(cfg.action.torque_limit_scale_range),[1.,1.])
            self.assertEqual(dict(cfg.observation),{})
            for name in BODIES[robot][1:]:
                self.assertTrue(any(re.match(p,name) for p in cfg.command.required_motion_body_patterns))

    def test_features_translation_invariance(self):
        pos=np.zeros((4,2,3));pos[:,1,0]=1;pos[:,:,2]=.5
        rot=np.broadcast_to(np.eye(3),(4,2,3,3));vel=np.zeros_like(pos)
        original=heading_features(pos,rot,vel)
        moved=pos+np.array([20,-4,0])
        np.testing.assert_allclose(original,heading_features(moved,rot,vel),atol=1e-6)

    def test_command_windows_do_not_cross_motion_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); shard=root/'train/s0';shard.mkdir(parents=True)
            (shard/'motions.json').write_text(json.dumps([{},{}]))
            for source in ['human','g1','roban']:
                np.save(shard/f'{source}_command.npy',np.array([[1],[2],[10],[20]],dtype=np.float16))
            schema=dict(offsets=[-1,0,1],splits={'train':[dict(path='train/s0',starts=[0,2],ends=[2,4],frames=4)]})
            (root/'dataset.json').write_text(json.dumps(schema));(root/'READY').write_text(sha256(root/'dataset.json'))
            bank=CommandBank(root,'train','cpu')
            values=bank.gather(torch.tensor([0,1]),torch.tensor([1,0]))['human']
            torch.testing.assert_close(values,torch.tensor([[1,2,2,1,1,0],[10,10,20,0,1,1]],dtype=torch.float32))


if __name__=='__main__':
    unittest.main()
