"""Audit actual teacher continuation or new E+ learner against its exact parent."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.learning.modules.xwbc_incremental import load_checkpoint_safe,equal_tree,tensor_digest
from active_adaptation.utils.xwbc_data import sha256


def audit(run,source,kind):
    old=load_checkpoint_safe(source)
    end=load_checkpoint_safe(run/'checkpoint_final.pt')
    cfg=json.loads((run/'config.json').read_text())
    rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    assert end['config']==cfg and end['constructor']==old['constructor']
    for key in ('format','context','dataset_digest','controls'):assert end[key]==old[key],key
    assert end['dataset_digest']==sha256(Path(cfg['dataset'])/'dataset.json')
    assert all(math.isfinite(v) for row in rows for v in row.values())
    assert all(torch.isfinite(v).all() for v in end['model'].values())
    if kind=='teacher':
        start=load_checkpoint_safe(run/'checkpoint_start.pt')
        assert old['stage']==start['stage']==end['stage']=='A'
        assert start['update']==old['update']==10000
        for key in ('model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng'):
            assert equal_tree(start[key],old[key]),'Continuation did not restore '+key
        assert end['parents']['sha256']==sha256(source)
        assert [x['update'] for x in rows]==list(range(10001,cfg['updates']+1))
        n=cfg['updates']-10000
        assert rows[-1]['incremental_transitions']==n*cfg['num_envs']*cfg['rollout_steps']
        steps_per_update=cfg['epochs']*math.ceil(cfg['num_envs']*cfg['rollout_steps']/cfg['minibatch_size'])
        initial_steps={k:int(v['step']) for k,v in old['optimizer']['state'].items()}
        assert all(int(v['step'])==initial_steps[k]+n*steps_per_update for k,v in end['optimizer']['state'].items())
        steps=n*steps_per_update;transitions=rows[-1]['incremental_transitions']
    else:
        assert old['stage']==end['stage']=='E' and old['update']==2300
        assert end['constructor']['action_decoder']['steps']==8
        assert end['parents']['initialization_stage']=='E' and end['parents']['sha256']==sha256(source)
        assert end['parents']['anchor']['sha256']==sha256(source)
        init=json.loads((run/'initialization.json').read_text())
        assert init['optimizer_state_count']==0 and all(x==0 for x in init['replay_sizes'].values())
        assert init['decoder_rng_devices']==[] and init['flow_steps']==init['anchor_flow_steps']==8
        initial=torch.load(run/'initial_model.pt',map_location='cpu',weights_only=True)
        assert equal_tree(initial,old['model'])
        for key,value in end['model'].items():
            if not key.startswith(('encoders.','proprio.','backbones.','heads.')):assert torch.equal(value,old['model'][key]),key
        for record in end['parents']['teachers'].values():assert sha256(record['checkpoint'])==record['sha256']
        assert [x['update'] for x in rows]==list(range(1,cfg['updates']+1))
        steps=cfg['updates']*cfg['supervised_steps'];transitions=cfg['updates']*cfg['num_envs']*cfg['rollout_steps']*2
        assert rows[-1]['optimizer_steps']==steps and rows[-1]['transitions']==transitions
        assert rows[-1]['replay_samples']==steps*cfg['batch_size']*2
        assert all(int(v['step'])==steps for v in end['optimizer']['state'].values())
        assert all(x['seen']==cfg['updates']*cfg['num_envs']*cfg['rollout_steps'] for x in end['replay'].values())
    assert end['update']==cfg['updates']
    for name,h in json.loads((run/'source_hashes.json').read_text()).items():assert sha256(run/'code'/name)==h
    return dict(passed=True,kind=kind,update=end['update'],additional_optimizer_steps=steps,
                additional_transitions=transitions,parent_sha256=sha256(source),
                checkpoint_sha256=sha256(run/'checkpoint_final.pt'),model_tensors_sha256=tensor_digest(end['model']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--kind',choices=['teacher','student'],required=True)
    args=p.parse_args();torch.set_num_threads(1)
    result=audit(args.run,args.source,args.kind)
    with (args.run/'incremental_audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
