"""Deterministic, source-family grouped RefineTrack data; never overwrites inputs."""
import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'dataset/roban/main/train_50000_v1'
SEED = 20260927

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def stable(key):
    return int.from_bytes(hashlib.sha256(f'{SEED}|{key}'.encode()).digest()[:8], 'big')

def family(path):
    # Conservatively group every performer/mirror suffix after __, without
    # claiming that naming establishes exact kinematic equivalence.
    p = Path(path)
    return str(p.parent / re.sub(r'__.*$', '', p.stem))

def read_arrays(path):
    schema = json.loads((path / '_tensordict/meta.json').read_text())
    return {k: np.memmap(path / '_tensordict' / (k + '.memmap'), mode='r', dtype=np.float16, shape=schema[k]['shape']) for k in ('root_pos_w','root_quat_w','joint_pos')}

def write_dataset(path, arrays, labels, meta):
    path.mkdir(parents=True, exist_ok=False)
    (path / '_tensordict').mkdir()
    schema = json.loads((SOURCE / '_tensordict/meta.json').read_text())
    for k, a in arrays.items():
        a.astype(np.float16).tofile(path / '_tensordict' / (k + '.memmap'))
        schema[k]['shape'] = list(a.shape)
    schema['shape'] = [len(arrays['joint_pos'])]
    dump(path / '_tensordict/meta.json', schema)
    dump(path / 'meta_motion.json', meta)
    dump(path / 'id_label.json', labels)

def corrupt(q, scale, width, joints, kind, severity, rng, limits):
    n = len(q); out = q.astype(np.float32).copy()
    w = max(1, min(n // 4, int(width * (2 ** (severity - 1)))))
    start = int(rng.integers(0, max(1, n - w + 1)))
    amp = scale[joints] * (0.5, 1., 2.)[severity-1]
    z = rng.normal(size=(w, len(joints)))
    if kind == 'offset' or kind == 'jump':
        z[:] = rng.choice([-1., 1.], size=len(joints))
    elif kind == 'burst':
        z *= np.sin(np.linspace(0, np.pi, w))[:, None]
    elif kind == 'correlated_noise':
        common = rng.normal(size=(w, 1))
        z = (z + common) / np.sqrt(2)
        rho = np.exp(-1 / max(width, 1))
        for t in range(1, w):
            z[t] = rho*z[t-1] + np.sqrt(1-rho*rho)*z[t]
    elif kind == 'drift':
        z = np.linspace(0, 1, w)[:,None] * rng.choice([-1.,1.],size=(1,len(joints)))
    z /= np.maximum(np.sqrt(np.mean(z*z, axis=0)), 1e-8)
    delta = np.clip(z, -3, 3) * amp
    ix = np.ix_(np.arange(start,start+w), joints)
    if kind == 'hold':
        out[ix] = q[start,joints][None,:]
    elif kind == 'seam':
        donor = max(0,start-w)
        out[ix] = q[np.ix_(np.arange(donor,donor+w),joints)]
    else:
        out[ix] += delta
    unclipped = out.copy()
    out = np.clip(out, limits[:,0], limits[:,1]).astype(np.float16)
    diff = out.astype(np.float32)-q.astype(np.float32)
    event_diff = diff[ix]
    return out, dict(start=start, length=w, kind=kind, severity=f'S{severity}', joints=joints,
        clipped_elements=int(np.count_nonzero(unclipped != np.clip(unclipped,limits[:,0],limits[:,1]))),
        event_rms=float(np.sqrt(np.mean(event_diff**2))), event_abs_p95=float(np.quantile(abs(event_diff),.95)),
        event_peak=float(abs(event_diff).max()), changed_elements=int(np.count_nonzero(diff)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'dataset/roban/refinetrack_v01/r2');args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); begin=time.time()
    meta=json.loads((SOURCE/'meta_motion.json').read_text());labels=json.loads((SOURCE/'id_label.json').read_text())
    selection=json.loads((SOURCE/'selection_manifest.json').read_text())['sources']
    # Union naming families with byte-identical original source files.
    parent={family(r['relative_path']):family(r['relative_path']) for r in selection}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    hashes={}
    for r in selection:
        f=family(r['relative_path']); h=r['sha256_before']
        if h in hashes:
            a,b=find(f),find(hashes[h]);parent[max(a,b)]=min(a,b)
        else: hashes[h]=f
    rows=[]
    for r in selection:
        f=find(family(r['relative_path']));u=stable('split|'+f)/2**64
        split='train' if u<.1 else 'validation' if u<.11 else 'test' if u<.12 else 'unused'
        rows.append(dict(source_path=r['relative_path'],source_sha256=r['sha256_before'],family_id=f,split=split))
    by_source={r['source_path']:r for r in rows}
    dump(out/'source_manifest.json',dict(seed=SEED,definition='hash family threshold: train [0,.1), validation [.1,.11), test [.11,.12); original 50000 selected sources, including conversion failures',sources=rows))
    arrays=read_arrays(SOURCE)
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(ROOT/'active_adaptation/assets/roban/mjcf/biped_s17_verified_handball_fixedhead.xml'))
    limits=np.array([model.joint(n).range for n in meta['joint_names']])
    # Round limits inward on the FP16 storage grid, so FK and runtime clamp agree.
    lo=limits[:,0].astype(np.float16);hi=limits[:,1].astype(np.float16)
    lo=np.where(lo<limits[:,0],np.nextafter(lo,np.float16(np.inf)),lo)
    hi=np.where(hi>limits[:,1],np.nextafter(hi,np.float16(-np.inf)),hi)
    limits=np.stack([lo,hi],axis=1).astype(np.float32)
    reports={}; clean_sets={}
    for split in ['train','validation','test']:
        ids=[i for i,r in enumerate(labels) if by_source[r['source_path']]['split']==split]
        lengths=[meta['ends'][i]-meta['starts'][i] for i in ids]; ends=np.cumsum(lengths).tolist();starts=[0]+ends[:-1]
        selected_labels=[labels[i] for i in ids]
        selected_meta=dict(joint_names=meta['joint_names'],body_names=meta['body_names'],starts=starts,ends=ends,info={})
        a={k:np.concatenate([v[meta['starts'][i]:meta['ends'][i]] for i in ids]) for k,v in arrays.items()}
        original=a['joint_pos'].astype(np.float32);a['joint_pos']=np.clip(original,limits[:,0],limits[:,1]).astype(np.float16)
        assert all(np.isfinite(v).all() for v in a.values())
        write_dataset(out/split/'clean',a,selected_labels,selected_meta)
        manifests=[{'original_motion_id':i,**labels[i],**by_source[labels[i]['source_path']]} for i in ids]
        dump(out/f'{split}_manifest.json',manifests)
        reports[split]=dict(selected_sources=sum(r['split']==split for r in rows),valid_sources=len({r['source_path'] for r in selected_labels}),segments=len(ids),frames=sum(lengths),families=len({r['family_id'] for r in manifests}),source_directory_coverage=dict(Counter(str(Path(r['source_path']).parent) for r in manifests)),canonical_clipped_elements=int(np.count_nonzero(original!=a['joint_pos'])))
        clean_sets[split]=(a,selected_labels,selected_meta)
    train,_,tm=clean_sets['train'];deltas=[];ranges=[]
    for s,e in zip(tm['starts'],tm['ends']):
        q=train['joint_pos'][s:e].astype(np.float32);deltas.append(abs(np.diff(q,axis=0)));ranges.append(np.quantile(q,.95,axis=0)-np.quantile(q,.05,axis=0))
    ds=np.concatenate(deltas);scale=np.quantile(ds,.95,axis=0);amplitude=np.median(ranges,axis=0)
    names=meta['joint_names']
    for i,n in enumerate(names):
        if '_l' in n:
            j=names.index(n.replace('_l','_r'));scale[i]=scale[j]=np.quantile(ds[:,[i,j]],.95)
    groups={'waist':[0],'left_hip':[1,2,3],'left_knee':[4],'left_ankle':[5,6],'right_hip':[7,8,9],'right_knee':[10],'right_ankle':[11,12],'left_arm':[13,14,15,16],'right_arm':[17,18,19,20]}
    widths={g:max(1,int(np.rint(np.median(amplitude[j]/np.maximum(scale[j],1e-8))))) for g,j in groups.items()}
    protocol=dict(version='joint_v1_r1',seed=SEED,numpy=np.__version__,scale=scale.tolist(),amplitude=amplitude.tolist(),groups=groups,widths=widths,severity_factors=[0,.5,1,2],train_mixture=[1/3]*3,train_types=['jitter','offset','burst','correlated_noise'],heldout_types=['jump','drift','hold','seam'],root_corruption=False,limits=limits.tolist(),train_statistics='all train frames; mirror pooled delta P95; median per-motion P95-P05',generator_sha256=sha(__file__))
    dump(out/'corruption_protocol.json',protocol)
    events=[];pair_checks=[]
    for split,(a,lab,m) in clean_sets.items():
        severities=[1,2] if split=='train' else [1,2,3]
        for sev in severities:
            raw={k:v for k,v in a.items()};raw['joint_pos']=a['joint_pos'].copy()
            types=protocol['train_types'] # held-out cases are built separately by evaluator preparation
            for i,(s,e) in enumerate(zip(m['starts'],m['ends'])):
                identity=json.dumps(lab[i],sort_keys=True); key=f'joint_v1_r1|{split}|{identity}|S{sev}|0';rng=np.random.Generator(np.random.PCG64(stable(key)))
                kind=types[int(rng.integers(len(types)))];group=list(groups)[int(rng.integers(len(groups)))];j=groups[group]
                q,event=corrupt(a['joint_pos'][s:e],scale,widths[group],j,kind,sev,rng,limits)
                raw['joint_pos'][s:e]=q;events.append(dict(split=split,motion_id=i,group=group,seed=stable(key),**event))
            path=out/split/f'raw_S{sev}';write_dataset(path,raw,lab,m)
            for k in ['root_pos_w','root_quat_w']:
                assert sha(path/'_tensordict'/f'{k}.memmap')==sha(out/split/'clean/_tensordict'/f'{k}.memmap')
            assert sha(path/'id_label.json')==sha(out/split/'clean/id_label.json')
            pair_checks.append(dict(split=split,severity=sev,identity=True,root_bitwise_equal=True,finite=bool(np.isfinite(raw['joint_pos']).all()),different=bool(np.any(raw['joint_pos']!=a['joint_pos']))))
        for condition in ['C0','C1']:
            base=str((out/split/'clean').relative_to(ROOT/'dataset'))
            gs=[dict(name=f'{condition}_S{s}',teacher_mem_path=base,student_mem_path=base if condition=='C0' or s==0 else str((out/split/f'raw_S{s}').relative_to(ROOT/'dataset')),weight=1/3) for s in range(3)]
            dump(out/f'{split}_{condition}_groups.json',gs)
    dump(out/'corruption_events.json',events)
    checksums={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    report=dict(status='DATA_CHECKS_PASS',splits=reports,pair_checks=pair_checks,hashes=checksums,seconds=time.time()-begin,bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),limitations=['Source-family naming and exact source hash grouping; no exhaustive semantic near-duplicate certification.','Held-out corruption types not yet materialized.','Canonical S0 bridge evaluation pending.'])
    dump(out/'full_report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['hashes','splits']},indent=2));print({k:{x:y for x,y in v.items() if x!='source_directory_coverage'} for k,v in reports.items()})
if __name__=='__main__':main()
