"""Check FSQ reload, unchanged teachers, frozen parameters and exact learning budget."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.learning.xwbc_staged import load_model, context_contract
from active_adaptation.learning.modules.student_latent_quantizer import StudentLatentQuantizer
from active_adaptation.utils.xwbc_data import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    torch.set_num_threads(1)
    cfg = json.loads((run/'config.json').read_text())
    model, state = load_model(run/'checkpoint_final.pt', 'cpu')
    assert state['stage'] in ('B', 'E')
    assert state['constructor']['student_fsq'] == cfg['student_fsq']
    assert cfg['student_fsq']['enabled'] and state['config'] == cfg
    assert state['context'] == context_contract(cfg)
    assert state['dataset_digest'] == sha256(Path(cfg['dataset'])/'dataset.json')
    quantizers = [m for m in model.modules() if isinstance(m, StudentLatentQuantizer)]
    assert len(quantizers) == 3 and not any(list(m.parameters()) for m in quantizers)
    for source in ('human', 'g1', 'roban'):
        z = model.encoders['shared'][source](torch.randn(16, model.constructor['command_dims'][source]))
        assert z.shape == (16, cfg['token_dim'])
        torch.testing.assert_close(z, z.round(), rtol=0, atol=0)
        assert z.min() >= -4 and z.max() <= 4
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    rows = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    assert [r['update'] for r in rows] == list(range(1, cfg['updates']+1))
    assert state['update'] == cfg['updates']
    assert all(math.isfinite(v) for row in rows for v in row.values())
    assert rows[-1]['transitions'] == cfg['updates']*cfg['num_envs']*cfg['rollout_steps']*2
    steps = cfg['updates']*cfg['supervised_steps'] if state['stage'] == 'B' else rows[-1]['optimizer_steps']
    assert all(int(v['step']) == steps for v in state['optimizer']['state'].values())
    teachers = state['parents'] if state['stage'] == 'B' else state['parents']['teachers']
    for robot in ('g1', 'roban'):
        record = teachers[robot]
        assert sha256(record['checkpoint']) == record['sha256']
        teacher, _ = load_model(record['checkpoint'], 'cpu')
        assert not any(isinstance(m, StudentLatentQuantizer) for m in teacher.modules())
    if state['stage'] == 'B':
        initial = torch.load(run/'initial_model.pt', map_location='cpu', weights_only=False)
        allowed = ('encoders.shared.human.', 'backbones.', 'proprio.', 'heads.')
        for name, value in state['model'].items():
            if not name.startswith(allowed):
                assert torch.equal(value, initial[name]), name
    for name, h in json.loads((run/'source_hashes.json').read_text()).items():
        assert sha256(run/'code'/name) == h
    result = dict(passed=True, stage=state['stage'], student_fsq=cfg['student_fsq'],
                  updates=state['update'], optimizer_steps=steps, transitions=rows[-1]['transitions'],
                  checkpoint_sha256=sha256(run/'checkpoint_final.pt'), teacher_weights='unchanged')
    with (run/'fsq_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
