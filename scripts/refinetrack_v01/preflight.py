"""Fail closed on drift, split leakage, and same-length paired identity errors."""
import json
from pathlib import Path
from scripts.refinetrack_v01.build_data import sha,read_arrays

def check_pair(clean,raw):
    clean,raw=Path(clean),Path(raw)
    for name in ['id_label.json','meta_motion.json']:
        if sha(clean/name)!=sha(raw/name):raise ValueError(f'paired identity/schema mismatch: {name}')
    for name in ['root_pos_w.memmap','root_quat_w.memmap']:
        if sha(clean/'_tensordict'/name)!=sha(raw/'_tensordict'/name):raise ValueError(f'paired root mismatch: {name}')

def check_data(root):
    root=Path(root);report=json.loads((root/'full_report.json').read_text())
    assert report['status']=='DATA_CHECKS_PASS'
    for name,expected in report['hashes'].items():
        if sha(root/name)!=expected:raise ValueError(f'dataset changed: {name}')
    heldout=root/'heldout_report.json'
    if heldout.exists():
        for name,expected in json.loads(heldout.read_text())['hashes'].items():
            if sha(root/name)!=expected:raise ValueError(f'heldout data changed: {name}')
    seen_family={};seen_source={}
    for split in ['train','validation','test']:
        for row in json.loads((root/f'{split}_manifest.json').read_text()):
            for key,seen in [('family_id',seen_family),('source_path',seen_source)]:
                value=row[key]
                if value in seen and seen[value]!=split:raise ValueError(f'{key} split leak: {value}')
                seen[value]=split
        for raw in sorted((root/split).glob('raw_S*')):check_pair(root/split/'clean',raw)
    return {'status':'PASS','files_hashed':len(report['hashes']),'families':len(seen_family),'sources':len(seen_source)}
