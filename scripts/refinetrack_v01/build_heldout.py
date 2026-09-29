"""Materialize predeclared test-only held-out corruption without evaluating it."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.refinetrack_v01.build_data import ROOT,read_arrays,write_dataset,corrupt,stable,dump,sha
from scripts.refinetrack_v01.preflight import check_pair

def main():
    root=ROOT/'dataset/roban/refinetrack_v01/r2';p=json.loads((root/'corruption_protocol.json').read_text());clean=root/'test/clean';a=read_arrays(clean);m=json.loads((clean/'meta_motion.json').read_text());labels=json.loads((clean/'id_label.json').read_text());events=[]
    for sev in [1,2,3]:
        raw=dict(a);raw['joint_pos']=a['joint_pos'].copy()
        for i,(s,e) in enumerate(zip(m['starts'],m['ends'])):
            seed=stable(f'heldout|test|{json.dumps(labels[i],sort_keys=True)}|S{sev}|0');rng=np.random.default_rng(seed)
            kind=p['heldout_types'][int(rng.integers(4))];g=list(p['groups'])[int(rng.integers(len(p['groups'])))];j=p['groups'][g]
            q,event=corrupt(a['joint_pos'][s:e],np.array(p['scale']),p['widths'][g],j,kind,sev,rng,np.array(p['limits']));raw['joint_pos'][s:e]=q;events.append(dict(motion_id=i,seed=seed,group=g,**event))
        target=root/'test'/f'raw_heldout_S{sev}';write_dataset(target,raw,labels,m);check_pair(clean,target)
    dump(root/'heldout_events.json',events)
    dump(root/'heldout_report.json',dict(status='PASS',generator_sha256=sha(__file__),hashes={str(f.relative_to(root)):sha(f) for directory in (root/'test').glob('raw_heldout*') for f in sorted(directory.rglob('*')) if f.is_file()},test_evaluated=False))
if __name__=='__main__':main()
