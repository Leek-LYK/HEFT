"""Exercise the training logger setup without constructing a simulator."""
import ast
import datetime
import logging
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf


def check_disabled_wandb_checkpoint_roundtrip(tmp_path):
    tree = ast.parse(Path('scripts/train.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'need_logging' for t in n.targets))
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rollout_policy' for t in n.targets))
    setup = compile(ast.Module(body=main.body[start:end], type_ignores=[]), 'train_logging_setup', 'exec')
    cfg = OmegaConf.create({'wandb': {'mode': 'disabled'}, 'exp_name': 'local-test', 'algo': {'train_every': 32}})
    policy = torch.nn.Linear(3, 2)
    env = SimpleNamespace(max_episode_length=1000, reward_spec=SimpleNamespace(keys=lambda *a: []), device='cpu', state_dict=lambda: {'value': torch.ones(1)})
    init = Mock(side_effect=AssertionError('W&B must not start'))
    context = dict(aa=SimpleNamespace(is_main_process=lambda: True), cfg=cfg, datetime=datetime,
        hydra=SimpleNamespace(core=SimpleNamespace(hydra_config=SimpleNamespace(HydraConfig=SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))))))),
        init_wandb_run=init, setproctitle=lambda n: None, os=os, OmegaConf=OmegaConf,
        policy=policy, env=env, vecnorm=torch.nn.Linear(3, 3), logging=logging,
        EpisodeStats=lambda *a, **kw: None, OrderedDict=OrderedDict, torch=torch, save_interval=150)
    exec(setup, context)
    for name in ('checkpoint_150', 'checkpoint_final'):
        context['save'](policy, name)
        state = torch.load(tmp_path / f'{name}.pt', weights_only=False)
        restored = torch.nn.Linear(3, 2)
        restored.load_state_dict(state['policy'], strict=True)
        sample = torch.randn(5, 3)
        torch.testing.assert_close(policy(sample), restored(sample))
        assert 'wandb' not in state
        assert {'policy', 'cfg', 'env', 'vecnorm'} <= state.keys()
        assert not (tmp_path / f'{name}.pt.tmp').exists()
    assert (tmp_path / 'cfg.yaml').exists()
    init.assert_not_called()
    context['aa'] = SimpleNamespace(is_main_process=lambda: False)
    exec(setup, context)
    assert not context['need_logging']


class LocalLoggingTest(unittest.TestCase):
    def test_disabled_wandb(self):
        with tempfile.TemporaryDirectory() as directory:
            check_disabled_wandb_checkpoint_roundtrip(Path(directory))


if __name__ == '__main__':
    unittest.main()
