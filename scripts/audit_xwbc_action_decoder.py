"""Independently verify decoder identity, private RNG, endpoint reload, and latency."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from active_adaptation.learning.xwbc_staged import load_model
from active_adaptation.learning.modules.xwbc_staged import student_mean, student_features
from active_adaptation.utils.xwbc_data import sha256


def audit(run, device='cpu'):
    torch.set_num_threads(4)
    model, state = load_model(run/'checkpoint_final.pt', device)
    cfg = json.loads((run/'config.json').read_text())
    assert model.constructor['action_decoder'] == cfg['action_decoder']
    assert model.heads.kind == cfg['action_decoder']['kind']
    assert model.heads.steps == 4
    if model.heads.kind == 'flow':
        assert state['decoder_rng'], 'Missing trained noise generator state'
    if state['stage'] == 'B':
        initial = torch.load(run/'initial_model.pt', map_location='cpu', weights_only=False)
        allowed = ('encoders.shared.human.', 'proprio.', 'backbones.', 'heads.')
        assert all(torch.equal(v.cpu(), initial[k]) for k,v in model.state_dict().items() if not k.startswith(allowed))
    else:
        parent = torch.load(state['parents']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        assert parent['constructor'] == state['constructor']
        assert state['parents']['sha256'] == sha256(state['parents']['checkpoint'])
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    timing = {}; sensitivity = {}
    with torch.no_grad():
        for robot in ('g1', 'roban'):
            for batch in (1, cfg['num_envs']):
                prop = torch.randn(batch, model.constructor['proprio_dims'][robot], device=device)
                cmd = torch.randn(batch, model.constructor['command_dims']['human'], device=device)
                fn = lambda: student_mean(model, robot, prop, cmd, 'human')
                first = fn()
                if model.heads.kind == 'flow' and batch == 1:
                    h=student_features(model,robot,prop,cmd,'human')
                    generator=torch.Generator(device=device).manual_seed(42)
                    samples=torch.stack([model.heads.predict(robot,h,initial=torch.randn(first.shape,device=device,generator=generator)) for _ in range(16)])
                    sensitivity[robot]=dict(noise_initializations=16,endpoint_std_rms=float(samples.std(0).square().mean().sqrt()),
                                            zero_vs_sample_mean_rms=float((first-samples.mean(0)).square().mean().sqrt()),synthetic_inputs=True)
                assert first.shape == (batch, model.constructor['action_dims'][robot])
                assert torch.equal(first, fn()) and torch.isfinite(first).all()
                for _ in range(5): fn()
                times = []
                for _ in range(30):
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    start = time.perf_counter()
                    fn()
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                timing[f'{robot}/batch_{batch}'] = dict(mean_ms=sum(times)/len(times), p95_ms=sorted(times)[int(.95*(len(times)-1))])
    return dict(passed=True, stage=state['stage'], kind=model.heads.kind, steps=model.heads.steps,
                checkpoint_sha256=sha256(run/'checkpoint_final.pt'),
                total_parameters=sum(p.numel() for p in model.parameters()),
                decoder_parameters=sum(p.numel() for p in model.heads.parameters()),
                initialization_sensitivity=sensitivity,
                latency=dict(device=device, synthetic_inputs=True, includes_encoders_and_backbone=True, measurements=timing))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    result=audit(args.run,args.device)
    with (args.run/'action_decoder_audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
