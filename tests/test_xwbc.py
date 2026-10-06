import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

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
            original_digest = sha256(root/'dataset.json')
            wide = CommandBank(root, 'train', 'cpu', offsets=[0, 20, -8])
            values = wide.gather(torch.tensor([0, 1]), torch.tensor([1, 0]))['human']
            torch.testing.assert_close(values, torch.tensor([[2,2,1,1,0,0],[10,20,10,1,0,0]], dtype=torch.float32))
            self.assertEqual(sha256(root/'dataset.json'), original_digest)

    def test_heft_history_taps_wrap_and_partial_reset(self):
        from active_adaptation.envs.xwbc import XWBCPool
        from active_adaptation.utils.helpers import TensorRingBuffer
        pool = object.__new__(XWBCPool)
        pool.action_dim = 1
        pool.action_history = 8
        pool.history_steps = [0,1,2,3,4,8,12,16,20]
        pool._history_indices = torch.tensor(pool.history_steps)
        pool._state_history = TensorRingBuffer(2,21,8,device='cpu',dtype=torch.float32)
        actions = TensorRingBuffer(2,8,1,device='cpu',dtype=torch.float32)
        pool.env = SimpleNamespace(action_manager=SimpleNamespace(get_recent_action_obs=actions.recent))
        for t in range(30):
            pool._state_history.push(torch.tensor([[t]*8,[1000+t]*8],dtype=torch.float32))
            actions.push(torch.tensor([[t-1],[1000+t-1]],dtype=torch.float32))
        observation = pool.history_observation()
        torch.testing.assert_close(observation[0,:72].reshape(9,8)[:,0],
                                   torch.tensor([29,28,27,26,25,21,17,13,9],dtype=torch.float32))
        torch.testing.assert_close(observation[0,72:],torch.arange(28,20,-1,dtype=torch.float32))
        pool.proprio = lambda: torch.full((2,9),99.)
        actions.reset(torch.tensor([0]))  # HEFT action-manager reset precedes history reset.
        pool._reset_history(torch.tensor([0]))
        reset = pool.history_observation()
        torch.testing.assert_close(reset[1],observation[1])
        torch.testing.assert_close(reset[0,:8],torch.full((8,),99.))
        self.assertTrue((reset[0,8:]==0).all())

    def test_old_encoder_constructor_and_new_token_capacity(self):
        old = self.make_policy()
        constructor = dict(old.constructor)
        constructor.pop('encoder_hidden')  # run01 checkpoint constructor
        restored = XWBCPolicy(**constructor)
        restored.load_state_dict(old.state_dict())
        inputs={s:torch.randn(4,n) for s,n in dict(human=5,g1=6,roban=7).items()}
        x=torch.randn(4,8); route=torch.arange(4)%3
        torch.testing.assert_close(old.actor('g1',x,route,inputs)[0].mean,
                                   restored.actor('g1',x,route,inputs)[0].mean,rtol=0,atol=0)
        wide=XWBCPolicy(dict(human=5,g1=6,roban=7),dict(g1=8,roban=9),dict(g1=3,roban=2),
                        token_dim=256,encoder_hidden=[512],hidden=[8])
        self.assertEqual(wide.tokens('g1',inputs).shape,(4,3,256))
        z=torch.randn(4,3,128)
        torch.testing.assert_close(alignment_loss(z),2*alignment_loss(z.repeat(1,1,2)))

    def test_wide_config_and_legacy_history_options(self):
        from active_adaptation.learning.xwbc import load_train_config, history_options
        root=Path(__file__).resolve().parents[1]
        old=load_train_config(root/'cfg/xwbc/train.yaml')
        wide=load_train_config(root/'cfg/xwbc/heft_context.yaml')
        self.assertEqual(old['updates'],2500)
        self.assertEqual(history_options(old),{})
        self.assertEqual(wide['updates'],10000)
        self.assertEqual(wide['token_dim'],256)
        self.assertEqual(wide['num_envs'],old['num_envs'])
        self.assertEqual(max(wide['command_offsets']),20)
        self.assertEqual(max(history_options(wide)['history_steps']),20)


if __name__=='__main__':
    unittest.main()
