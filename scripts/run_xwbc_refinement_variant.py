"""Run one controlled refinement variant, then evaluate its checkpoints concurrently."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-training-pid',type=int)
    parser.add_argument('--eval-iterations',type=int,nargs='+',default=[100,200,300])
    parser.add_argument('--test-num-envs',type=int,default=64)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    run=args.output.resolve()
    command=[sys.executable,'-u','-B','scripts/refine_xwbc_student.py','--config',args.config,'--output',str(run)]
    if args.wait_training_pid:
        while not (run/'result.json').exists():
            os.kill(args.wait_training_pid,0)
            stat=Path(f'/proc/{args.wait_training_pid}/stat').read_text().split(') ')[1].split()[0]
            if stat=='Z':
                raise RuntimeError('Training terminated without a result')
            time.sleep(15)
        result=json.loads((run/'result.json').read_text())
        if result['status']!='complete' or result['updates']<max(args.eval_iterations):
            raise RuntimeError('Incomplete training result')
    else:
        subprocess.run(command,check=True)
    workers={}
    try:
        for iteration in args.eval_iterations:
            cmd=[sys.executable,'-u','-B','scripts/xwbc_staged.py','evaluate','--checkpoint',str(run/f'checkpoint_{iteration}.pt'),
                 '--output',str(run/f'eval_val_{iteration}'),'--num-envs','64','--split','val']
            with (run/f'eval_val_{iteration}.console.log').open('xb') as log:
                workers[str(iteration)]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        pids={k:v.pid for k,v in workers.items()}
        with (run/'evaluation.launch.json').open('x') as f:
            json.dump(dict(pids=pids,started_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()),f,indent=2)
        subprocess.run([sys.executable,'-u','-B','scripts/finish_xwbc_refinement.py','--run-dir',str(run),
                        '--evaluation-pids',json.dumps(pids),'--iterations',*[str(i) for i in args.eval_iterations],
                        '--test-num-envs',str(args.test_num_envs)],check=True)
        for worker in workers.values():
            if worker.wait()!=0:
                raise RuntimeError('Evaluation failed')
    finally:
        # Preserve live evaluators if analysis fails; their PIDs and outputs remain inspectable.
        pass


if __name__=='__main__':
    main()
