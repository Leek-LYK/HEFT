"""Independent CPU audit of refinement lineage, frozen parameters and run artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    baseline=torch.load(args.baseline,map_location='cpu',weights_only=False)
    path=args.checkpoint or args.run/'checkpoint_final.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert baseline['stage'] in ('B','C','B_T') and state['stage'] in ('E','E_T')
    assert state['parents'].get('initialization_stage', 'C') == baseline['stage'].removesuffix('_T')
    for key in ('format','constructor','context','controls','dataset_digest'):
        assert state[key]==baseline[key],key
    assert state['parents']['sha256']==digest(args.baseline)
    cfg=json.loads((args.run/'config.json').read_text())
    assert state['config']==cfg
    assert state['dataset_digest']==digest(Path(cfg['dataset'])/'dataset.json')
    assert state['model'].keys()==baseline['model'].keys()
    allowed=('backbones.','proprio.','heads.')+ (('encoders.',) if cfg['train_sources'] else ())
    changed=[];unchanged=[]
    for name,value in state['model'].items():
        assert value.shape==baseline['model'][name].shape,name
        assert torch.isfinite(value).all(),name
        if torch.equal(value,baseline['model'][name]):
            unchanged.append(name)
        else:
            assert name.startswith(allowed),name
            changed.append(name)
    assert changed
    for name,expected in json.loads((args.run/'source_hashes.json').read_text()).items():
        assert digest(args.run/'code'/name)==expected,name
    for robot,record in state['parents']['teachers'].items():
        assert digest(record['checkpoint'])==record['sha256'],robot
    rows=[json.loads(line) for line in (args.run/'metrics.jsonl').read_text().splitlines()]
    assert [r['update'] for r in rows]==list(range(1,cfg['updates']+1))
    assert all(math.isfinite(v) for r in rows for v in r.values())
    assert rows[-1]['transitions']==cfg['updates']*cfg['num_envs']*cfg['rollout_steps']*2
    if cfg.get('parent_stage') == 'B':
        assert state['parents']['anchor']['sha256'] == digest(args.baseline)
        expected_steps = expected_samples = 0
        for row in rows:
            block = next(b for b in cfg['budget_schedule'] if row['update'] <= b['until_update'])
            expected_steps += block['supervised_steps']
            expected_samples += block['supervised_steps']*block['batch_size']*2
            assert row['optimizer_steps'] == expected_steps
            assert row['replay_samples'] == expected_samples
            assert row['supervised_steps'] == block['supervised_steps']
            assert row['batch_size'] == block['batch_size']
    assert 0<state['update']<=cfg['updates']
    assert rows[state['update']-1]['update']==state['update']
    result=dict(passed=True,checkpoint=str(path.resolve()),checkpoint_sha256=digest(path),
                baseline_sha256=state['parents']['sha256'],updates=state['update'],
                transitions=rows[state['update']-1]['transitions'],run_total_transitions=rows[-1]['transitions'],
                train_sources=cfg['train_sources'],
                changed_parameters=changed,unchanged_parameters=unchanged,
                architecture_context_control_data='identical_to_parent',
                parent_teachers_and_code_snapshots='hashes_verified',metrics='finite_contiguous_complete')
    with (args.output or args.run/'independent_audit.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('changed_parameters','unchanged_parameters')},ensure_ascii=False))


if __name__=='__main__':
    main()
