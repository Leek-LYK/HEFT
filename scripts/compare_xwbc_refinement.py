"""Paired family bootstrap against C; never change or filter evaluation rows."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(baseline, candidate):
    summaries = [json.loads((p/'summary.json').read_text()) for p in (baseline,candidate)]
    for key in ('dataset_digest','context','split','motion_ids_digest','expected_clips'):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError('Incomparable '+key)
    if not all(s['full_scope'] for s in summaries):
        raise ValueError('Full split required')
    rows = []
    for directory in (baseline,candidate):
        records = [json.loads(x) for x in (directory/'clips.jsonl').read_text().splitlines()]
        keyed = {(r['source'],r['robot'],r['motion_id']):r for r in records}
        if len(keyed)!=len(records):
            raise ValueError('Duplicate evaluation rows')
        rows.append(keyed)
    if rows[0].keys()!=rows[1].keys():
        raise ValueError('Route/motion coverage differs')
    keys=sorted(rows[0])
    if any(rows[0][k]['family_id']!=rows[1][k]['family_id'] for k in keys):
        raise ValueError('Family mapping differs')
    rng=np.random.default_rng(20261005)
    routes=sorted(summaries[0]['matrix'])
    result=dict(baseline=str(baseline.resolve()),candidate=str(candidate.resolve()),
                split=summaries[0]['split'],baseline_sha256=summaries[0]['checkpoint_sha256'],
                candidate_sha256=summaries[1]['checkpoint_sha256'],records=len(keys),routes={})
    def interval(selected):
        families=sorted({rows[0][k]['family_id'] for k in selected})
        totals=np.zeros(len(families));counts=np.zeros(len(families))
        index={f:i for i,f in enumerate(families)}
        for k in selected:
            i=index[rows[0][k]['family_id']]
            totals[i]+=int(rows[1][k]['success'])-int(rows[0][k]['success'])
            counts[i]+=1
        draws=rng.integers(len(families),size=(10000,len(families)))
        samples=100*totals[draws].sum(1)/counts[draws].sum(1)
        return np.quantile(samples,[.025,.975]).tolist()
    for route in routes:
        b=summaries[0]['matrix'][route];c=summaries[1]['matrix'][route]
        s,r=route.split('->');selected=[k for k in keys if k[:2]==(s,r)]
        result['routes'][route]=dict(
            baseline_success=b['success_rate'],candidate_success=c['success_rate'],
            success_delta_pp=100*(c['success_rate']-b['success_rate']),
            success_delta_ci95_pp=interval(selected),
            completion_delta_pp=100*(c['completion']-b['completion']),
            root_relative_error_delta_m=c['mpkpe_root_relative_m']-b['mpkpe_root_relative_m'],
            world_error_delta_m=c['mpkpe_world_m']-b['mpkpe_world_m'],
            baseline_recoveries=b['recovery_clips'],candidate_recoveries=c['recovery_clips'])
    result['means']={metric:dict(baseline=float(np.mean([summaries[0]['matrix'][r][metric] for r in routes])),
                                candidate=float(np.mean([summaries[1]['matrix'][r][metric] for r in routes])))
                     for metric in ('success_rate','completion','mpkpe_root_relative_m','mpkpe_world_m')}
    result['pooled_success_delta_ci95_pp']=interval(keys)
    # Recompute completion and step-weighted errors for each paired family draw.
    # Errors keep their original route-specific denominators, including failed active steps.
    families=sorted({rows[0][k]['family_id'] for k in keys})
    family_index={f:i for i,f in enumerate(families)}
    route_index={r:i for i,r in enumerate(routes)}
    shape=(2,len(routes),len(families))
    count=np.zeros(shape);steps=np.zeros(shape);completion=np.zeros(shape)
    error_sums={m:np.zeros(shape) for m in ('mpkpe_root_relative_m','mpkpe_world_m')}
    for side in (0,1):
        for k,row in rows[side].items():
            i=route_index[k[0]+'->'+k[1]];j=family_index[row['family_id']]
            count[side,i,j]+=1;steps[side,i,j]+=row['steps']
            completion[side,i,j]+=row['completion']
            for metric,total in error_sums.items():
                total[side,i,j]+=row['errors'][metric]*row['steps']
    draws=rng.integers(len(families),size=(10000,len(families)))
    def paired_interval(numerator,denominator,scale=1.):
        boot=(numerator[:,:,draws].sum(-1)/np.maximum(denominator[:,:,draws].sum(-1),1)).mean(1)
        return np.quantile(scale*(boot[1]-boot[0]),[.025,.975]).tolist()
    result['mean_delta_family_ci95']={'completion_pp':paired_interval(completion,count,100),
        **{m:paired_interval(total,steps) for m,total in error_sums.items()}}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=compare(args.baseline,args.candidate)
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['means'],ensure_ascii=False))
