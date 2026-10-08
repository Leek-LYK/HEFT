import copy,unittest,tempfile
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import torch
from active_adaptation.learning.modules.xwbc_incremental import eight_step_origin,equal_tree,tensor_digest,load_checkpoint_safe
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.xwbc_staged import student_mean
from active_adaptation.learning.xwbc import load_train_config
from active_adaptation.learning.xwbc_staged import config_for
ROOT=Path(__file__).resolve().parents[1]


class IncrementalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def state(self):
        model=XWBCPolicy(dict(human=5,g1=6,roban=7),dict(g1=8,roban=9),dict(g1=3,roban=2),token_dim=4,hidden=[8],encoder_hidden=[8],
             student_fsq=dict(enabled=True),action_decoder=dict(kind='flow',width=16,steps=4,action_embed=4,time_embed=4,robot_embed=4))
        state=dict(stage='E',update=2300,constructor=model.constructor,model=model.state_dict(),
             config=dict(action_decoder=copy.deepcopy(model.constructor['action_decoder'])),optimizer={'old':1},replay={'old_labels':1},decoder_rng={'old':1})
        return model,state
    def test_conversion_changes_only_solver_and_discards_old_learning_state(self):
        m,s=self.state();before=copy.deepcopy(s);out=eight_step_origin(s,'source.pt','sha')
        self.assertTrue(equal_tree(s,before));self.assertTrue(equal_tree(s['model'],out['model']))
        expected=copy.deepcopy(s['constructor']);expected['action_decoder']['steps']=8
        self.assertEqual(out['constructor'],expected);self.assertIsNone(out['optimizer']);self.assertIsNone(out['replay']);self.assertEqual(out['decoder_rng'],{})
        converted=XWBCPolicy(**out['constructor']);converted.load_state_dict(out['model'])
        x,c=torch.randn(4,8),torch.randn(4,5);m.heads.steps=8
        torch.testing.assert_close(student_mean(m,'g1',x,c,'human'),student_mean(converted,'g1',x,c,'human'),rtol=0,atol=0)
        self.assertEqual(tensor_digest(s['model']),out['solver_conversion']['model_tensors_sha256'])
    def test_wrong_origin_or_already_changed_solver_rejected(self):
        _,s=self.state()
        for key,value in [('stage','B'),('update',2000)]:
            bad=copy.deepcopy(s);bad[key]=value
            with self.assertRaises(ValueError):eight_step_origin(bad,'x','x')
        bad=copy.deepcopy(s);bad['constructor']['action_decoder']['steps']=8
        with self.assertRaises(ValueError):eight_step_origin(bad,'x','x')
    def test_teacher_config_preserves_learning_recipe(self):
        base=config_for(ROOT/'cfg/xwbc/v02/staged.yaml','A',SimpleNamespace())
        new=config_for(ROOT/'cfg/xwbc/v02/teacher_resume_15k.yaml','A',SimpleNamespace())
        self.assertEqual(new['updates'],15000)
        for key in ['learning_rate','num_envs','rollout_steps','epochs','minibatch_size','clip','entropy_weight','value_weight','seed','hidden']:
            self.assertEqual(base[key],new[key],key)
    def test_incremental_student_budget_and_explicit_E_parent(self):
        cfg=load_train_config(ROOT/'cfg/xwbc/v02/refinement_flow_incremental.yaml')
        self.assertEqual(cfg['parent_stage'],'E');self.assertEqual(cfg['action_decoder']['steps'],8)
        self.assertEqual(cfg['updates']*cfg['num_envs']*cfg['rollout_steps']*2,32768000)
        self.assertEqual(cfg['updates']*cfg['supervised_steps'],16000)
        self.assertEqual(cfg['expected_student_update'],2300)
    def test_restricted_loader_roundtrips_numpy_rng_and_tensor_state(self):
        state=dict(model={'weight':torch.randn(3,4)},numpy_rng=np.random.get_state(),optimizer={'state':{0:{'step':torch.tensor(2.)}}})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'s.pt';torch.save(state,path)
            self.assertTrue(equal_tree(state,load_checkpoint_safe(path)))
    def test_restricted_loader_rejects_unregistered_types(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'s.pt';torch.save({'unknown':SimpleNamespace(value=1)},path)
            with self.assertRaises(Exception):load_checkpoint_safe(path)
    def test_tree_audit_catches_optimizer_or_rng_change(self):
        a=dict(x=torch.ones(3),seq=[torch.arange(4),{'step':torch.tensor(3.)}]);b=copy.deepcopy(a)
        self.assertTrue(equal_tree(a,b));b['seq'][1]['step'].add_(1);self.assertFalse(equal_tree(a,b))
if __name__=='__main__':unittest.main()
