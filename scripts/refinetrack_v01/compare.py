"""Paired case comparison with family-cluster bootstrap, never frame bootstrap."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.refinetrack_v01.build_data import dump,family

def compare(left,right):
    a=json.loads((left/'cases.json').read_text());b=json.loads((right/'cases.json').read_text())
    sa=json.loads((left/'summary.json').read_text());sb=json.loads((right/'summary.json').read_text())
    for key in ['split','severity','seed','recipe','manifest_sha256','protocol_sha256','physics']:
        assert sa[key]==sb[key],f'Case protocol mismatch: {key}'
    assert len(a)==len(b) and all(x['source']==y['source'] for x,y in zip(a,b))
    fam={}
    for i,row in enumerate(a):fam.setdefault(family(row['source']['source_path']),[]).append(i)
    rng=np.random.default_rng(20260927);clusters=list(fam.values());d=np.array([float(y['success'])-float(x['success']) for x,y in zip(a,b)])
    draws=[]
    for _ in range(2000):
        idx=np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))]);draws.append(d[idx].mean())
    common=[i for i,(x,y) in enumerate(zip(a,b)) if x['success'] and y['success']]
    result=dict(cases=len(a),families=len(fam),C0_success=sa['success_rate'],C1_success=sb['success_rate'],success_delta_C1_minus_C0=float(d.mean()),success_delta_family_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),common_success_count=len(common),metrics={})
    for name in a[0]['metrics']:
        result['metrics'][name]={}
        for subset,idx in [('all_pretermination',list(range(len(a)))),('common_success',common)]:
            result['metrics'][name][subset]={label:float(np.mean([rows[i]['metrics'][name] for i in idx])) if idx else None for label,rows in [('C0',a),('C1',b)]}
    result['own_success_metrics']={label:{name:float(np.mean([r['metrics'][name] for r in rows if r['success']])) if any(r['success'] for r in rows) else None for name in rows[0]['metrics']} for label,rows in [('C0',a),('C1',b)]}
    result['limitation']='Single training/evaluation seed; cluster CI measures source variation only. All-case errors have unequal survival durations.'
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();dump(a.output,compare(a.left,a.right))
