"""Parallel ours-HEFT ONNX evaluation using the existing AMP/sim2sim control path."""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from analyze_brain_root_relative_tracking import BODIES, INIT, SAMPLE

ROOT=Path(__file__).resolve().parents[1]
TEMPLATE=ROOT/'dataset/_audit/roban_ominisoma/brain_amp_heft_20260909/111_a_person_is_walking_forward_slowly/simulate.py'
POLICY=ROOT/'dataset/_audit/roban_ominisoma/deploy_our_heft_20260908/policy_project_heft_finetune.onnx'
FINISH='''
pos=np.asarray(relative_positions).reshape(-1,len(metric_names))
ori=np.asarray(relative_orientations).reshape(-1,len(metric_names))
qerr=np.asarray(joint_errors).reshape(-1,21)
assert np.isfinite(pos).all() and np.isfinite(ori).all() and np.isfinite(qerr).all()
success=report['playback_started'] and report['playback_complete'] and not report['fell']
if success: assert len(pos)==len(csv)-1,(len(pos),len(csv))
np.savez_compressed(OUT/'tracking_samples.npz',position_mm=pos,orientation_rad=ori,joint_error_rad=qerr,frame=np.asarray(metric_frames),bodies=np.asarray(metric_names))
metrics={'case':CASE_NAME,'success':success,'sample_count':len(pos),'motion_frames':len(csv),
 'joint_rmse_rad':float(np.sqrt(np.mean(qerr*qerr,axis=1)).mean()) if len(pos) else None,
 'joint_mae_rad':float(np.abs(qerr).mean()) if len(pos) else None,
 'root_relative_body_pos_mm':float(pos.mean()) if len(pos) else None,
 'root_relative_body_ori_rad':float(ori.mean()) if len(pos) else None,
 'policy_sha256':POLICY_HASH,'source_csv_sha256':report['source_csv_sha256'],
 'root_translation_range_m':np.ptp(csv[:,:3],axis=0).tolist(),
 'per_joint_mae_rad':np.abs(qerr).mean(0).tolist() if len(pos) else None}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\\n')
'''

def evaluate(csv,out,template,policy_hash):
    case=csv.stem.removesuffix('_heft_50hz');folder=out/case.split('_',1)[0]
    folder.mkdir(exist_ok=True)
    done=folder/'metrics.json'
    if done.exists():
        m=json.loads(done.read_text());assert m['policy_sha256']==policy_hash and m['source_csv_sha256']==hashlib.sha256(csv.read_bytes()).hexdigest();return m
    source=template
    start=source.index("sys.path.insert(0,str(ROOT/");end=source.index('csvpath=',start)
    source=source[:start]+f"ours_policy=Path({str(POLICY)!r})\nmodelpaths.append(str(ours_policy));sessions.append(ort.InferenceSession(str(ours_policy),sess_options=so,providers=['CPUExecutionProvider']))\n"+source[end:]
    start=source.index('csvpath=');end=source.index('\n',start)
    source=source[:start]+f"csvpath=Path({str(csv)!r});csv=np.loadtxt(csvpath,skiprows=1)"+source[end:]
    replacements={
        'OUT=Path(__file__).parent':f'OUT=Path({str(folder)!r})',
        'tracking=[];trace=[];video_origin=np.zeros(2);fallen=False':INIT+'\njoint_errors=[]\ntracking=[];trace=[];video_origin=np.zeros(2);fallen=False',
        '   tracking.append(':SAMPLE+'   joint_errors.append((data.qpos[qidx]-csv[frame,7:28]).copy())\n   tracking.append(',
    }
    for old,new in replacements.items():assert source.count(old)==1;source=source.replace(old,new)
    prefix=f'BODY_NAMES={BODIES!r}\nCASE_NAME={case!r}\nPOLICY_HASH={policy_hash!r}\n'
    script=folder/'evaluate.py';script.write_text(prefix+source+'\n'+FINISH)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    begin=time.monotonic()
    with (folder/'console.log').open('w') as log:
        p=subprocess.run([sys.executable,'-B',str(script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode: return {'case':case,'execution_error':True,'returncode':p.returncode,'log':str(folder/'console.log')}
    m=json.loads(done.read_text());m['wall_seconds']=time.monotonic()-begin
    done.write_text(json.dumps(m,indent=2)+'\n');return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--limit',type=int);a=p.parse_args()
    files=sorted(a.input_dir.resolve().glob('*.csv'));assert files
    if a.limit:files=files[:a.limit]
    ids=[x.stem.split('_',1)[0] for x in files];assert len(ids)==len(set(ids))
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    template=TEMPLATE.read_text();policy_hash=hashlib.sha256(POLICY.read_bytes()).hexdigest()
    provenance={'policy':str(POLICY),'policy_sha256':policy_hash,'template':str(TEMPLATE),'template_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'workers':a.workers,'motion_count':len(files),'backend':'ONNX Runtime CPU','success_definition':'playback_started and playback_complete and not fell; includes final hold','metric_window':'50Hz pre-integration samples; reference frames 0 through N-2; excludes AMP, transition, wait, final hold','bodies':BODIES}
    (out/'run_config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results=[];start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(evaluate,f,out,template,policy_hash):f for f in files}
        for future in concurrent.futures.as_completed(fs):
            try:r=future.result()
            except Exception as e:r={'case':fs[future].stem,'execution_error':True,'error':repr(e)}
            results.append(r)
            if len(results)%25==0 or len(results)==len(files):
                print(json.dumps({'done':len(results),'total':len(files),'success':sum(x.get('success',False) for x in results),'execution_errors':sum(x.get('execution_error',False) for x in results),'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    results.sort(key=lambda x:x['case']);(out/'batch_results.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(x.get('execution_error') for x in results):raise SystemExit(1)
if __name__=='__main__':main()
