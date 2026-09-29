"""Recompute every stored corruption event and require exact FP16 replay."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.refinetrack_v01.build_data import ROOT,read_arrays,corrupt,dump
root=ROOT/'dataset/roban/refinetrack_v01/r2';p=json.loads((root/'corruption_protocol.json').read_text());count=0
for heldout in [False,True]:
    events=json.loads((root/('heldout_events.json' if heldout else 'corruption_events.json')).read_text());cache={}
    for event in events:
        split='test' if heldout else event['split'];sev=int(event['severity'][1:]);key=(split,sev)
        if key not in cache:
            m=json.loads((root/split/'clean/meta_motion.json').read_text());clean=read_arrays(root/split/'clean')['joint_pos'];raw=read_arrays(root/split/(f'raw_heldout_S{sev}' if heldout else f'raw_S{sev}'))['joint_pos'];cache[key]=(m,clean,raw)
        m,clean,raw=cache[key];i=event['motion_id'];s,e=m['starts'][i],m['ends'][i]
        rng=np.random.default_rng(event['seed']);kinds=p['heldout_types'] if heldout else p['train_types'];kind=kinds[int(rng.integers(4))];group=list(p['groups'])[int(rng.integers(len(p['groups'])))];assert kind==event['kind'] and group==event['group']
        q,_=corrupt(clean[s:e],np.array(p['scale']),p['widths'][group],p['groups'][group],kind,sev,rng,np.array(p['limits']))
        assert np.array_equal(q,raw[s:e]),(split,sev,i);count+=1
report=dict(status='PASS',events_exactly_replayed=count,rng='PCG64',stored_dtype='float16',root_checked_by_preflight=True)
dump(ROOT/'科研/v0.1/evidence/corruption_replay.json',report);print(report)
