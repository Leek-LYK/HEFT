"""Run a matched validation-only matrix; no test selection or optimization."""
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.refinetrack_v01.build_data import ROOT
models={
'C0':ROOT/'dataset/_audit/refinetrack_v01/smoke_C0_finetune_20260928_01/checkpoint_final.pt',
'C1':ROOT/'dataset/_audit/refinetrack_v01/smoke_C1_finetune_20260928_01/checkpoint_final.pt',
'historical':ROOT/'dataset/_audit/roban_ominisoma/roban_pipeline_20260908_091456/finetune/checkpoint_final.pt'}
if __name__=='__main__':
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',MEMPATH=str(ROOT/'dataset'),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    for name,cp in models.items():
        for severity in ['S0','S1','S2']:
            output=ROOT/f'dataset/_audit/refinetrack_v01/validation_matrix_v2/{name}_{severity}'
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.with_suffix('.log').open('x') as log:
                subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/refinetrack_v01/evaluate.py'),'--checkpoint',str(cp),'--limit','16','--severity',severity,'--output',str(output)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(name,severity,'PASS',flush=True)
