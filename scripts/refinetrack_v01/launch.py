"""Bounded smoke launcher. Formal launch remains gated on complete acceptance."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.refinetrack_v01.build_data import ROOT,sha,dump
DATA=ROOT/'dataset/roban/refinetrack_v01/r2'
PARENT=ROOT/'dataset/_audit/roban_ominisoma/roban_teacher_50000_r50000_v3/wandb/wandb/offline-run-20260905_141941-roban_teacher_50000_r50000_v3/files/checkpoint_final.pt'

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['C0','C1'],required=True);p.add_argument('--stage',choices=['train','adapt','finetune'],required=True);p.add_argument('--gpu',choices=['3','4','7'],default='3');p.add_argument('--parent',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke-rollouts',type=int,default=3);p.add_argument('--formal',action='store_true');a=p.parse_args()
    assert 1<=a.smoke_rollouts<=10
    contract_path=ROOT/'科研/v0.1/experiment_contract.json'
    if a.formal:
        contract=json.loads(contract_path.read_text())
        assert contract['status']=='FROZEN_SMOKE_PASSED', 'Formal launch blocked until smoke acceptance'
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==contract['git_head'], 'Git HEAD drift requires revision'
        for name,expected in contract['implementation_hashes'].items():
            assert sha(ROOT/name)==expected, f'Implementation drift: {name}'
        assert sha(DATA/'full_report.json')==contract['dataset']['report_sha256']
    free=int(subprocess.check_output(['nvidia-smi','--id='+a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=23000, f'GPU {a.gpu} has only {free} MiB free; wait for >=23000 MiB, do not terminate other jobs'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='roban/RefineTrack/v0.1'
    from scripts.refinetrack_v01.preflight import check_data
    check_data(DATA)
    parent=a.parent or PARENT;assert parent.is_file()
    if a.stage!='train':assert a.parent is not None
    if a.formal:
        if a.stage=='train':
            assert parent.resolve()==PARENT.resolve() and sha(parent)==contract['parent_checkpoints']['teacher_sha256']
        else:
            previous=json.loads((parent.parent/'run_record.json').read_text())
            assert previous['kind']=='formal' and previous['condition']==a.condition
            assert previous['stage']=={'adapt':'train','finetune':'adapt'}[a.stage]
            assert previous['checkpoint_sha256']==sha(parent)
            assert previous['contract_sha256']==sha(contract_path)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    budget={'train':10000,'adapt':1000,'finetune':1000}[a.stage];horizon=16 if a.stage=='adapt' else 32
    gs=DATA/f'train_{a.condition}_groups.json';groups=json.loads(gs.read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,MEMPATH=str(ROOT/'dataset'),WANDB_MODE='disabled',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/mpl-refinetrack',OMP_NUM_THREADS='8')
    args=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/train.py'),'task=tracking',f'+exp=roban_{a.stage}','task.num_envs=4096',f'task.profile.command.dataset.groups={yaml.safe_dump(groups,default_flow_style=True,width=1000000,sort_keys=False).strip()}', '+task.profile.command.dataset.resample_on_reset=true','task.profile.command.student_motion_randomization.enable=false','task.profile.randomization.window_cap_hand_load.enabled=false','task.action.boot_protect=false',f'algo.train_every={horizon}',f'total_frames={4096*horizon*budget}','start_iter=0',f'stop_after_rollouts={a.smoke_rollouts}',f'checkpoint_path={parent.resolve()}','seed=0','save_interval=1000','wandb.mode=disabled','tensorboard.enabled=true',f'tensorboard.log_dir={a.output}/tensorboard','train_record.enabled=false','headless=true','app.headless=true','app.enable_cameras=false','eval_render=false',f'hydra.run.dir={a.output}','hydra.job.chdir=false','hydra.output_subdir=null','+refinetrack.enabled=true',f'+refinetrack.condition={a.condition}']
    diff=subprocess.check_output(['git','diff','--binary'],cwd=ROOT);(a.output/'working_tree.patch').write_bytes(diff)
    source_hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'scripts/refinetrack_v01').glob('*.py'))}
    if a.formal:args=[x for x in args if not x.startswith('stop_after_rollouts=')]
    record=dict(contract_sha256=sha(contract_path),source_hashes=source_hashes,diff_sha256=sha(a.output/'working_tree.patch'),date=datetime.datetime.now().astimezone().isoformat(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),condition=a.condition,stage=a.stage,run_id=a.output.name,gpu=a.gpu,seed=0,num_envs=4096,horizon=horizon,planned_transitions=4096*horizon*(budget if a.formal else a.smoke_rollouts),native_budget_rollouts=budget,manifest_sha256=sha(DATA/'train_manifest.json'),protocol_sha256=sha(DATA/'corruption_protocol.json'),parent=str(parent.resolve()),parent_sha256=sha(parent),output=str(a.output),wandb_id=None,command=args,kind='formal' if a.formal else 'smoke',selected_checkpoint=None)
    (a.output/'experiment_contract.json').write_bytes(contract_path.read_bytes())
    dump(a.output/'run_record.json',record)
    with (a.output/'console.log').open('w') as f:
        code=subprocess.call(args,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    record['exit_code']=code
    cp=a.output/'checkpoint_final.pt'
    if code==0 and cp.is_file():record['selected_checkpoint']=str(cp);record['checkpoint_sha256']=sha(cp)
    if (a.output/'cfg.yaml').exists():record['config_sha256']=sha(a.output/'cfg.yaml')
    dump(a.output/'run_record.json',record)
    print(json.dumps(record,indent=2));sys.exit(code)
if __name__=='__main__':main()
