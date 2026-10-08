"""Two prespecified A->B->E decoder experiments, independent gates and full comparisons."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from scripts.run_xwbc_no_c_ablation import save, sha, checked_reference
from scripts.compare_xwbc_refinement import compare
from active_adaptation.learning.xwbc import load_train_config
from active_adaptation.learning.modules.xwbc import XWBCPolicy


def design_check():
    reference=ROOT/'logs/v0.2/fsq_20261007/run01/E_FSQ/checkpoint_2300.pt'
    old=torch.load(reference,map_location='cpu',weights_only=False,mmap=True)
    models={};cfgs={}
    for variant in ('flow','decoder_mlp'):
        cfg=load_train_config(ROOT/f'cfg/xwbc/v02/staged_{variant}.yaml');cfgs[variant]=cfg
        constructor=dict(old['constructor'],action_decoder=cfg['action_decoder'])
        torch.manual_seed(0)
        models[variant]=XWBCPolicy(**constructor)
    a,b=models.values()
    for k,v in a.state_dict().items():
        if not k.startswith('heads.'): assert torch.equal(v,b.state_dict()[k]),k
    ca,cb=[sum(p.numel() for p in m.heads.parameters()) for m in models.values()]
    assert abs(ca-cb)/ca < .01
    for variant,cfg in cfgs.items():
        base=load_train_config(ROOT/'cfg/xwbc/v02/staged_fsq.yaml')
        stripped=dict(cfg);stripped.pop('action_decoder');assert base==stripped
        old_e=load_train_config(ROOT/'cfg/xwbc/v02/refinement_fsq.yaml')
        new_e=load_train_config(ROOT/f'cfg/xwbc/v02/refinement_{variant}.yaml')
        assert new_e.pop('action_decoder')==cfg['action_decoder']
        for metadata in ('method','evaluation_reference'):
            old_e.pop(metadata);new_e.pop(metadata)
        assert old_e==new_e
    return dict(passed=True,decoder_parameters=dict(flow=ca,decoder_mlp=cb),
                parameter_difference_fraction=abs(ca-cb)/ca,common_initial_weights='bit_identical',
                total_parameters={v:sum(p.numel() for p in m.parameters()) for v,m in models.items()},
                variants=cfgs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();os.chdir(ROOT);torch.set_num_threads(1)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    design=design_check();save(out/'design_check.json',design)
    reference=ROOT/'logs/v0.2/fsq_20261007/run01/E_FSQ'
    for split in ('val','test'):checked_reference(reference/f'eval_{split}_2300',reference/'checkpoint_2300.pt')
    save(out/'protocol.json',dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        variants=['flow','decoder_mlp'],reference=str(reference),reference_sha256=sha(reference/'checkpoint_2300.pt'),
        per_variant=dict(B_updates=2000,E_updates=2300,transitions=140902400,optimizer_steps=132800,replay_samples=534118400),
        seed=0,primary_flow_steps=4,flow_initialization='zero_in_rollout_and_eval',secondary_val_steps=[1,2,8],
        equal_compute=False,teacher_target='frozen_A_mean_at_student_visited_state',
        E_anchor='own_frozen_starting_B_same_source_endpoint',anchor_weight=.1,
        test_used_for_selection=False,matching_noise_rng='private_checkpointed_generator',
        config_sha256={str(p.relative_to(ROOT)):sha(p) for v in ('flow','decoder_mlp') for p in [ROOT/f'cfg/xwbc/v02/staged_{v}.yaml',ROOT/f'cfg/xwbc/v02/refinement_{v}.yaml']}))
    workers={};results={}
    for variant,label in [('flow','FLOW'),('decoder_mlp','MLP')]:
        command=[sys.executable,'-u','-B','scripts/run_xwbc_fsq.py','--output',str(out/variant),
                 '--config',f'cfg/xwbc/v02/staged_{variant}.yaml','--refinement-config',f'cfg/xwbc/v02/refinement_{variant}.yaml',
                 '--reference',str(reference),'--student-reference',str(reference.parent/'student_human/eval_val'),'--label',label]
        with (out/f'{variant}.driver.log').open('x') as log:
            workers[variant]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    save(out/'workers.json',{k:p.pid for k,p in workers.items()})
    # A failed variant never cancels the other experimental arm.
    pending=set(workers)
    while pending:
        for variant in sorted(pending.copy()):
            code=workers[variant].poll()
            if code is not None:
                results[variant]=dict(exit_code=code,status='complete' if code==0 else 'stopped')
                save(out/f'{variant}.exit.json',results[variant]);pending.remove(variant)
        if pending:time.sleep(15)
    # Compare actually saved initial tensors, not just a construction demonstration.
    initial=[torch.load(out/v/'student_human/initial_model.pt',map_location='cpu',weights_only=False) for v in ('flow','decoder_mlp') if (out/v/'student_human/initial_model.pt').exists()]
    if len(initial)==2:
        same=all(torch.equal(v,initial[1][k]) for k,v in initial[0].items() if not k.startswith('heads.'))
        save(out/'initialization_audit.json',dict(passed=same,common_initial_weights='bit_identical' if same else 'different'))
        if not same:raise RuntimeError('Actual initial common weights differed')
    if all(x['exit_code']==0 for x in results.values()):
        for split in ('val','test'):
            save(out/f'flow_vs_mlp_{split}.json',compare(out/'decoder_mlp/E_MLP'/f'eval_{split}_2300',out/'flow/E_FLOW'/f'eval_{split}_2300'))
    save(out/'pipeline_result.json',dict(status='complete' if all(x['exit_code']==0 for x in results.values()) else 'finished_with_stopped_variant',variants=results,test_used_for_selection=False))


if __name__=='__main__':main()
