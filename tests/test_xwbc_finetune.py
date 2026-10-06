"""Behavioral checks for D warmup, frozen sources, trust region and resumption."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import frozen_digest
from active_adaptation.learning.xwbc_finetune import (
    BalancedSourcePool, configure_finetune, actor_mean_digest, update_finetune,
    restore_finetune, validate_finetune_config, rollout_metrics)
from active_adaptation.envs.xwbc_staged import StagedPool
from active_adaptation.learning.xwbc_staged import checkpoint, gate_decision, config_for


class FinetuneTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11); torch.set_num_threads(1)
        self.cfg = dict(num_envs=12, updates=4, rollout_steps=2, critic_warmup_updates=1,
                        action_std=.1, learning_rate=2e-5, critic_learning_rate=1e-4,
                        epochs=2, minibatch_size=6, clip=.1, value_weight=.5,
                        anchor_weight=.1, target_kl=100., max_grad_norm=1.,
                        checkpoint_interval=1, seed=0, command_offsets=[0], history_steps=[0], action_history=1)
        self.model = XWBCPolicy(dict(human=5,g1=6,roban=7),dict(g1=8,roban=9),dict(g1=3,roban=2),
                                token_dim=4,hidden=[8],encoder_hidden=[8])
        self.anchor = copy.deepcopy(self.model).requires_grad_(False)
        self.optimizer,self.actor,self.critic = configure_finetune(self.model,self.cfg)
        commands = {s:torch.randn(12,n) for s,n in dict(human=5,g1=6,roban=7).items()}
        self.bank = SimpleNamespace(device='cpu',gather=lambda ids,frame:{s:c[ids] for s,c in commands.items()})
        self.batches = {}
        with torch.no_grad():
            for robot,dim,adim in [('g1',8,3),('roban',9,2)]:
                proprio=torch.randn(12,dim);privileged=torch.randn(12,2*adim+9)
                source=torch.arange(12)%3;dist,_=self.model.actor(robot,proprio,source,commands)
                action=dist.sample();value=self.model.value(robot,proprio,privileged)
                self.batches[robot]=dict(proprio=proprio,privileged=privileged,source=source,
                    motion=torch.arange(12),frame=torch.zeros(12,dtype=torch.long),action=action,
                    old_mean=dist.mean.clone(),logp=dist.log_prob(action).sum(-1),value=value,returns=value+torch.linspace(.2,1.2,12),
                    advantage=torch.linspace(-1,1,12))

    def test_configuration_preserves_mean_and_sets_small_fixed_exploration(self):
        self.assertEqual(actor_mean_digest(self.anchor),actor_mean_digest(self.model))
        names={n for n,p in self.model.named_parameters() if p.requires_grad}
        self.assertTrue(all(n.startswith(('proprio.','backbones.','heads.','critics.')) for n in names))
        self.assertEqual({g['name']:g['lr'] for g in self.optimizer.param_groups},{'actor':2e-5,'critic':1e-4})
        for p in self.model.log_std.values():
            self.assertFalse(p.requires_grad);torch.testing.assert_close(p.exp(),torch.full_like(p,.1))

    def test_warmup_changes_critics_only(self):
        before=actor_mean_digest(self.model);critic={k:v.clone() for k,v in self.model.critics.state_dict().items()}
        m=update_finetune(self.model,self.anchor,self.optimizer,self.actor,self.critic,self.bank,self.batches,self.cfg,True)
        self.assertEqual(actor_mean_digest(self.model),before)
        self.assertTrue(any(not torch.equal(v,critic[k]) for k,v in self.model.critics.state_dict().items()))
        self.assertEqual(m['actor_gradient_norm'],0.)
        self.assertTrue(all(not self.optimizer.state[p] for p in self.actor))

    def test_ppo_updates_executor_and_keeps_source_encoders_and_anchor_exact(self):
        frozen=frozen_digest(self.model);before=actor_mean_digest(self.model)
        anchor={k:v.clone() for k,v in self.anchor.state_dict().items()}
        m=update_finetune(self.model,self.anchor,self.optimizer,self.actor,self.critic,self.bank,self.batches,self.cfg,False)
        self.assertNotEqual(actor_mean_digest(self.model),before)
        self.assertEqual(frozen_digest(self.model),frozen)
        self.assertGreater(m['actor_gradient_norm'],0.)
        self.assertTrue(all(torch.equal(v,anchor[k]) for k,v in self.anchor.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.anchor.parameters()))

    def test_excessive_initial_kl_prevents_optimizer_step(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        self.cfg['target_kl']=.01
        self.batches['roban']['old_mean']+=.2
        with self.assertRaisesRegex(RuntimeError,'no update'):
            update_finetune(self.model,self.anchor,self.optimizer,self.actor,self.critic,self.bank,self.batches,self.cfg,False)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.model.state_dict().items()))

    def test_post_step_backtracking_bounds_applied_kl(self):
        self.cfg.update(target_kl=.0001,max_backtracks=12,epochs=1,minibatch_size=12)
        self.optimizer.param_groups[0]['lr']=.01
        frozen=frozen_digest(self.model)
        m=update_finetune(self.model,self.anchor,self.optimizer,self.actor,self.critic,self.bank,self.batches,self.cfg,False)
        self.assertGreater(m['max_proposed_kl'],self.cfg['target_kl'])
        self.assertGreater(m['backtracks'],0)
        self.assertLessEqual(m['max_post_step_kl'],self.cfg['target_kl'])
        self.assertEqual(frozen_digest(self.model),frozen)

    def test_rejected_actor_step_restores_weights_and_optimizer_state(self):
        self.cfg.update(target_kl=1e-12,max_backtracks=0,epochs=1,minibatch_size=12)
        self.optimizer.param_groups[0]['lr']=.01
        before=actor_mean_digest(self.model)
        m=update_finetune(self.model,self.anchor,self.optimizer,self.actor,self.critic,self.bank,self.batches,self.cfg,False)
        self.assertEqual(m['actor_optimizer_steps'],0)
        self.assertEqual(actor_mean_digest(self.model),before)
        self.assertTrue(all(p not in self.optimizer.state for p in self.actor))
        self.assertTrue(any(p in self.optimizer.state for p in self.critic))
        self.assertLessEqual(m['max_post_step_kl'],self.cfg['target_kl'])

    def test_balanced_sources_survive_partial_reset(self):
        pool=object.__new__(BalancedSourcePool);pool.source=torch.full((1024,),-1,dtype=torch.long)
        def fake_reset(obj,ids):obj.source[ids]=99
        with patch.object(StagedPool,'reset',fake_reset):
            pool.reset(torch.arange(1024))
            self.assertEqual(torch.bincount(pool.source).tolist(),[342,341,341])
            original=pool.source.clone();pool.reset(torch.tensor([1,5,1000]))
            torch.testing.assert_close(pool.source,original,rtol=0,atol=0)

    def test_checkpoint_resume_preserves_optimizer_and_rejects_anchor_changes(self):
        update_finetune(self.model,self.anchor,self.optimizer,self.actor,self.critic,self.bank,self.batches,self.cfg,False)
        parents={'sha256':'C-anchor'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'D.pt'
            checkpoint(path,self.model,self.optimizer,'D',2,self.cfg,'data',parents,{})
            restored=copy.deepcopy(self.anchor);opt,_,_=configure_finetune(restored,self.cfg)
            target={**self.cfg,'updates':6}
            self.assertEqual(restore_finetune(SimpleNamespace(resume=path),restored,opt,target,'data',parents),2)
            for k,v in self.model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            self.assertTrue(opt.state)
            with self.assertRaisesRegex(ValueError,'anchor changed'):
                restore_finetune(SimpleNamespace(resume=path),restored,opt,target,'data',{'sha256':'wrong'})
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                restore_finetune(SimpleNamespace(resume=path),restored,opt,{**target,'action_std':.2},'data',parents)

    def test_d_gate_uses_each_matching_C_route(self):
        m=dict(clips=500,success_rate=.96,completion=.98,recovery_clips=0,mpkpe_root_relative_m=.04,mpkpe_world_m=.1)
        routes=[s+'->'+r for s in ['human','g1','roban'] for r in ['g1','roban']]
        bases={r:copy.deepcopy(m) for r in routes};bases['human->roban']['success_rate']=.90
        summary=dict(stage='D',robot=None,split='val',full_scope=True,expected_clips=500,
                     matrix=copy.deepcopy(bases),baselines=bases,
                     acceptance=dict(max_recovery_clip_fraction=.01,student_success_drop=.05,
                                     student_completion_drop=.05,student_relative_error_increase_m=.02))
        self.assertTrue(gate_decision(summary)['passed'])
        summary['matrix']['g1->roban']['success_rate']=.89
        self.assertIn('g1->roban:success_drop',gate_decision(summary)['issues'])

    def test_inherited_D_config_and_budget_validation(self):
        cfg=config_for('cfg/xwbc/v02/finetune.yaml','D',SimpleNamespace())
        validate_finetune_config(cfg)
        self.assertEqual(cfg['command_offsets'],[0,1,2,4,8,12,16,20,-1,-2,-4,-8])
        self.assertEqual(cfg['updates']-cfg['critic_warmup_updates'],500)
        for changed in [dict(num_envs=2),dict(action_std=1.),dict(updates=100)]:
            with self.assertRaises(ValueError):validate_finetune_config({**cfg,**changed})

    def test_rollout_success_logging_distinguishes_recovery_from_completion(self):
        b=dict(source=torch.tensor([0,1,2]),reward=torch.ones(3),done=torch.ones(3,dtype=torch.bool),
               completed=torch.tensor([True,True,False]),recovered=torch.tensor([False,True,False]),
               terminated=torch.tensor([False,True,True]),truncated=torch.tensor([True,False,False]))
        pool=SimpleNamespace(env=SimpleNamespace(nonfinite_state_recovery_count=1,reward_groups=[]))
        m=rollout_metrics({'g1':b},{'g1':pool})
        self.assertEqual(m['g1/human/train_episode_success_rate'],1.)
        self.assertEqual(m['g1/g1/train_episode_success_rate'],0.)
        self.assertEqual(m['g1/roban/train_episode_success_rate'],0.)

if __name__=='__main__':unittest.main()
