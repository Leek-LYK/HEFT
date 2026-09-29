"""Verify actual checkpoint updates and freeze sets across smoke stages."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.refinetrack_v01.build_data import ROOT,dump,sha

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    return a==b

def audit(path):
    r=json.loads((path/'run_record.json').read_text());assert r['exit_code']==0
    old=torch.load(r['parent'],map_location='cpu',weights_only=False);new=torch.load(path/'checkpoint_final.pt',map_location='cpu',weights_only=False)
    def finite(value):
        if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        return True
    assert finite(new['policy']) and finite(new['vecnorm']), 'Non-finite checkpoint'
    a,b=old['policy'],new['policy'];stage=r['stage']
    changes={k:not equal(a[k],b[k]) for k in ['actor_teacher','actor_student','encoder_priv','adapt_module','critic']}
    expected={'train':{'actor_teacher','encoder_priv','adapt_module','critic'},'adapt':{'adapt_module'},'finetune':{'actor_student','critic'}}[stage]
    # Native parent=train load copies Teacher to Student before any updates.
    if a['last_phase']=='train':changes['actor_student']=not equal(a['actor_teacher'],b['actor_student'])
    assert {k for k,v in changes.items() if v}==expected,(stage,changes)
    if stage!='train':assert equal(old['vecnorm'],new['vecnorm'])
    rows=[json.loads(x) for x in (path/'refinetrack_iterations.jsonl').read_text().splitlines()]
    assert len(rows)==3
    return dict(run=str(path),stage=stage,condition=r['condition'],status='PASS',module_updates=changes,vecnorm_changed=not equal(old['vecnorm'],new['vecnorm']),actual_transitions=sum(x['transitions'] for x in rows),steady_rollout_transitions_s=sum(x['rollout_transitions_per_s'] for x in rows[1:])/2,steady_total_s=sum(x['rollout_s']+x['training_s'] for x in rows[1:])/2,peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rows),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in rows),checkpoint_sha256=sha(path/'checkpoint_final.pt'))
if __name__=='__main__':
    rows=[audit(Path(p)) for p in sys.argv[1:]];dump(ROOT/'科研/v0.1/evidence/smoke_stage_audit.json',rows);print(json.dumps(rows,indent=2))
