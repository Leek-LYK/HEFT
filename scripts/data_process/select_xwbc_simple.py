"""Select paired, simple motions with explicit lexical and geometric criteria."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from active_adaptation.utils.xwbc_data import ROOT, validate_manifest

BANNED=re.compile(r'jump|kick|flip|crawl|kneel|cartwheel|handstand|roll|fall|injur|stair|climb|sit|chair|lie|lay|heavy|somersault|furniture|carry|backward')
PATTERNS={
 'run':re.compile(r'(^|_)(run|running|jog|jogging)(_|$)'),
 'walk':re.compile(r'(^|_)(walk|walking)(_|$)'),
 'upper':re.compile(r'(^|_)(wave|waving|clap|clapping|gesture|gesturing|talk|talking|point|pointing|nod|shrug|idle)(_|$)'),
 'shallow_squat':re.compile(r'(^|_)(squat|squatting)(_|$)')}

def digest(s):return hashlib.sha256(s.encode()).hexdigest()

def category(key):
 name=Path(key).name.split('__')[0].lower()
 if key.split('/')[0] not in ('main','squat') or BANNED.search(name):return None
 if key.startswith('squat/') or PATTERNS['shallow_squat'].search(name):return 'shallow_squat'
 for c in ('run','walk','upper'):
  if PATTERNS[c].search(name):return c
 return None

def examine(row):
 try:
  with np.load(ROOT/row['paths']['human'],allow_pickle=False) as z:
   h=z['motion'];names=z['joint_names'].tolist();fps=int(z['fps'])
  with np.load(ROOT/row['paths']['roban'],allow_pickle=False) as z:
   rj=z['joint_pos'];rfps=int(z['fps'].reshape(-1)[0]);rp=z['body_pos_w'][:,0]
  p=ROOT/row['paths']['g1']
  with p.open() as f:cols=f.readline().strip().split(',')
  g=np.loadtxt(p,delimiter=',',skiprows=1,ndmin=2)
  gj=np.deg2rad(g[:,7:]);gp=g[:,1:4]/100
  if not 100<=len(h)<=1500:raise ValueError('length_outside_2_to_30_seconds')
  if len(h)!=len(g) or len(h)!=len(rj) or fps!=50 or rfps!=50:raise ValueError('pairing_or_fps')
  if not all(np.isfinite(v).all() for v in (h,g,rj,rp)):raise ValueError('nonfinite')
  pos=h[...,:3];hip=pos[:,names.index('Hips')];chest=pos[:,names.index('Chest')]
  lengths=[]
  for side in ['Left','Right']:
   thigh,shin,foot=[pos[:,names.index(side+x)] for x in ['Leg','Shin','Foot']]
   lengths.append(np.linalg.norm(thigh-shin,axis=-1)+np.linalg.norm(shin-foot,axis=-1))
  leg_length=float(np.median(np.stack(lengths)))
  height_ratio=float(hip[:,2].min()/leg_length)
  torso=chest-hip
  tilt=float(np.rad2deg(np.arccos(np.clip(torso[:,2]/np.linalg.norm(torso,axis=-1),-1,1))).max())
  speed=max(float(np.linalg.norm(np.gradient(v,.02,axis=0),axis=-1).max()) for v in [hip,gp,rp])
  joint_speed=max(float(np.abs(np.gradient(v,.02,axis=0)).max()) for v in [gj,rj])
  knee=max(float(np.abs(g[:,[cols.index('left_knee_joint_dof'),cols.index('right_knee_joint_dof')]]*np.pi/180).max()),
           float(np.abs(rj[:,[15,16]]).max()))
  metrics=dict(frames=len(h),min_pelvis_height_over_leg_length=height_ratio,max_torso_tilt_deg=tilt,
               max_root_speed_mps=speed,max_joint_speed_radps=joint_speed,max_knee_angle_rad=knee)
  if height_ratio<.72:raise ValueError('deep_lowering')
  if tilt>55:raise ValueError('large_torso_tilt')
  if speed>4.5:raise ValueError('fast_root')
  if joint_speed>20:raise ValueError('joint_speed_spike')
  if row['category'] in ('upper','shallow_squat') and knee>1.2:raise ValueError('deep_knee_bend')
  if row['category']=='upper' and speed>1.2:raise ValueError('upper_body_root_travel')
  return dict(**row,quality=metrics)
 except (ValueError,KeyError,OSError,IndexError) as e:
  return dict(**row,rejection=str(e))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
 p.add_argument('--workers',type=int,default=2);p.add_argument('--scan-per-category',type=int,default=6000)
 a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 roots=dict(human=Path('dataset/Human_22d_50hz'),g1=Path('dataset/g1_50hz'),roban=Path('dataset/roban_50hz'))
 ext=dict(human='.npz',g1='.csv',roban='.npz')
 inv={s:{v.relative_to(ROOT/r).with_suffix('').as_posix() for v in (ROOT/r).rglob('*'+ext[s])} for s,r in roots.items()}
 candidates=defaultdict(list)
 for key in sorted(set.intersection(*inv.values())):
  c=category(key)
  if c is None:continue
  family=re.sub(r'_\d+$','',Path(key).name.split('__')[0]).casefold()
  bucket=int(digest('31:'+family)[:16],16)%100
  split='train' if bucket<80 else 'val' if bucket<90 else 'test'
  candidates[c].append(dict(motion_id=key,family_id=family,category=c,split=split,
                          paths={s:str(r/(key+ext[s])) for s,r in roots.items()}))
 scan=[r for c,items in sorted(candidates.items()) for r in sorted(items,key=lambda r:digest('simple:'+r['motion_id']))[:a.scan_per_category]]
 accepted=[];reasons=Counter()
 audit=out.with_suffix('.audit.jsonl')
 with audit.open('x') as log,ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,result in enumerate(pool.map(examine,scan,chunksize=8)):
   log.write(json.dumps(result,ensure_ascii=False)+'\n')
   if 'rejection' in result:reasons[result['rejection']]+=1
   else:accepted.append(result)
   if (i+1)%1000==0:print(f'audited={i+1}/{len(scan)} accepted={len(accepted)}',flush=True)
 selected=[];coverage={}
 weights=dict(walk=.5,run=.25,upper=.2,shallow_squat=.05)
 for split,total in [('train',4000),('val',500),('test',500)]:
  bins={c:[r for r in accepted if r['split']==split and r['category']==c] for c in weights}
  if sum(map(len,bins.values()))<total:raise ValueError(f'{split} has insufficient eligible motions; inspect {audit}')
  quota={c:min(len(bins[c]),round(total*w)) for c,w in weights.items()}
  while sum(quota.values())<total:
   options=[c for c in weights if quota[c]<len(bins[c])]
   c=min(options,key=lambda c:quota[c]/weights[c]);quota[c]+=1
  for c,items in bins.items():
   groups=defaultdict(list)
   for row in items:groups[row['family_id']].append(row)
   ranked=[]
   for f,members in groups.items():
    for rank,row in enumerate(sorted(members,key=lambda r:digest(r['motion_id']))):ranked.append((rank,digest(f),row))
   selected.extend(row for _,_,row in sorted(ranked,key=lambda x:(x[0],x[1]))[:quota[c]])
  coverage[split]=quota
 validate_manifest(selected)
 assert len(selected)==5000
 with out.open('x') as f:
  for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 report=dict(count=5000,split_seed=31,coverage=coverage,scanned=len(scan),accepted=len(accepted),
             rejection_counts=dict(reasons),lexical_candidates={c:len(v) for c,v in candidates.items()},
             family_rule='stem before __, remove numeric take, casefold; same global split seed as v0.1',
             criteria=dict(min_frames=100,max_frames=1500,min_height_ratio=.72,max_tilt_deg=55,
                           max_speed_mps=4.5,max_joint_speed_radps=20,max_shallow_knee_rad=1.2,max_upper_speed_mps=1.2),
             manifest_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
 out.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
