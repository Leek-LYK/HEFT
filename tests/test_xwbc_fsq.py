import copy
from pathlib import Path
import tempfile
import unittest
import torch
from active_adaptation.learning.modules.xwbc import XWBCPolicy
from active_adaptation.learning.modules.student_latent_quantizer import StudentLatentQuantizer
from active_adaptation.learning.modules.xwbc_staged import configure_student, student_mean, frozen_digest, Reservoir
from active_adaptation.learning.xwbc_staged import load_model, PROTOCOL, validate_parent, context_contract
from active_adaptation.learning.xwbc import load_train_config
from scripts.refine_xwbc_student import configure, update

ROOT = Path(__file__).resolve().parents[1]
FSQ = dict(enabled=True, scale=4.499549865722656)
DIMS = dict(human=5, g1=6, roban=7)


class FSQTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def model(self, fsq=None):
        return XWBCPolicy(DIMS, dict(g1=8, roban=9), dict(g1=3, roban=2),
                          token_dim=4, hidden=[8], encoder_hidden=[8], student_fsq=fsq)

    def test_baseline_initialization_and_disabled_path_unchanged(self):
        old = self.model()
        torch.manual_seed(5)
        quantized = self.model(FSQ)
        torch.manual_seed(5)
        disabled = self.model(dict(enabled=False))
        self.assertEqual(old.state_dict().keys(), quantized.state_dict().keys())
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, quantized.state_dict()[key]), key)
        x, c = torch.randn(4, 8), torch.randn(4, 5)
        torch.testing.assert_close(student_mean(old, 'g1', x, c, 'human'),
                                   student_mean(disabled, 'g1', x, c, 'human'), rtol=0, atol=0)
        self.assertNotIn('student_fsq', old.constructor)

    def test_all_actor_paths_quantized_and_ste_reaches_encoder(self):
        model = self.model(FSQ)
        commands = {s:torch.randn(6, d) for s,d in DIMS.items()}
        self.assertEqual(sum(isinstance(m, StudentLatentQuantizer) for m in model.modules()), 3)
        self.assertFalse(any(isinstance(m, StudentLatentQuantizer) for m in model.proprio.modules()))
        for r, dim in [('g1', 8), ('roban', 9)]:
            x = torch.randn(6, dim)
            tokens = model.tokens(r, commands)
            torch.testing.assert_close(tokens, tokens.round(), rtol=0, atol=0)
            self.assertTrue(bool((tokens.abs() <= 4).all()))
            for i, s in enumerate(DIMS):
                action = student_mean(model, r, x, commands[s], s)
                dist, _ = model.actor(r, x, torch.full((6,), i), commands)
                torch.testing.assert_close(action, dist.mean)
                model.zero_grad()
                action.square().sum().backward()
                grads = [p.grad for p in model.encoders['shared'][s].parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_B_E_checkpoint_reload_retains_quantization(self):
        for fsq in (None, FSQ):
            model = self.model(fsq)
            x, c = torch.randn(4, 8), torch.randn(4, 5)
            for stage in ('B', 'E'):
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d)/'model.pt'
                    torch.save(dict(format=PROTOCOL, stage=stage, constructor=model.constructor,
                                    model=model.state_dict()), p)
                    loaded, _ = load_model(p, 'cpu')
                    self.assertEqual(loaded.constructor, model.constructor)
                    torch.testing.assert_close(student_mean(model, 'g1', x, c, 'human'),
                                               student_mean(loaded, 'g1', x, c, 'human'), rtol=0, atol=0)

    def test_fsq_parent_mismatch_rejected(self):
        cfg = load_train_config(ROOT/'cfg/xwbc/v02/staged_fsq.yaml')
        state = dict(stage='B', constructor=self.model().constructor)
        with self.assertRaisesRegex(ValueError, 'FSQ'):
            validate_parent(state, cfg, 'B')

    def test_real_configs_only_change_planned_factor(self):
        for baseline, new in [('staged', 'staged_fsq'), ('teacher_refinement_no_c', 'refinement_fsq')]:
            a = load_train_config(ROOT/f'cfg/xwbc/v02/{baseline}.yaml')
            b = load_train_config(ROOT/f'cfg/xwbc/v02/{new}.yaml')
            self.assertEqual(b.pop('student_fsq'), FSQ)
            for metadata in ('method', 'evaluation_reference'):
                a.pop(metadata, None)
                b.pop(metadata, None)
            self.assertEqual(a, b)

    def test_B_freezing_then_six_route_E_updates_through_fsq(self):
        model = self.model(FSQ)
        optimizer = torch.optim.Adam(configure_student(model, 'B'), lr=.01)
        before = frozen_digest(model)
        optimizer.zero_grad()
        student_mean(model, 'g1', torch.randn(4, 8), torch.randn(4, 5), 'human').sum().backward()
        optimizer.step()
        self.assertEqual(before, frozen_digest(model))
        anchor = copy.deepcopy(model).requires_grad_(False)
        initial = {k:v.clone() for k,v in model.state_dict().items()}
        optimizer = torch.optim.Adam(configure(model, True), lr=.003)
        before = frozen_digest(model)
        class Bank:
            device = 'cpu'
            def gather(self, motion, frame):
                return {s:torch.ones(len(motion), d) for s,d in DIMS.items()}
        replays = {}
        for r, dim, actions in [('g1', 8, 3), ('roban', 9, 2)]:
            replays[r] = Reservoir(16)
            replays[r].add(dict(proprio=torch.ones(16, dim), motion=torch.zeros(16, dtype=torch.long),
                                frame=torch.zeros(16, dtype=torch.long), target=torch.ones(16, actions)))
        update(model, anchor, optimizer, Bank(), replays,
               dict(supervised_steps=10, batch_size=8, anchor_weight=.1, max_grad_norm=1))
        self.assertEqual(before, frozen_digest(model))
        self.assertTrue(all(torch.equal(v, initial[k]) for k,v in anchor.state_dict().items()))
        for s in DIMS:
            self.assertTrue(any(not torch.equal(v, initial[k]) for k,v in model.state_dict().items()
                                if k.startswith('encoders.shared.'+s+'.')))


if __name__ == '__main__':
    unittest.main()
