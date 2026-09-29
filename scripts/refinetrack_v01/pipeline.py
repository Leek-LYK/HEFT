"""Launch isolated stages and follow the actual saved checkpoint index."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.refinetrack_v01.build_data import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['C0','C1'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',choices=['3','4','7'],required=True);p.add_argument('--formal',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);parent=None
    for stage in ['train','adapt','finetune']:
        stage_dir=a.output/stage
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/refinetrack_v01/launch.py'),'--condition',a.condition,'--stage',stage,'--gpu',a.gpu,'--output',str(stage_dir)]
        if a.formal:cmd+=['--formal']
        if parent:cmd+=['--parent',parent]
        subprocess.run(cmd,check=True)
        record=json.loads((stage_dir/'run_record.json').read_text());parent=record['selected_checkpoint']
        if not parent or not Path(parent).is_file():raise RuntimeError('No actual selected checkpoint; stopping chain')
if __name__=='__main__':main()
