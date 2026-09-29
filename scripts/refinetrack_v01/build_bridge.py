"""Preserve original reference processing on the exact validation sources."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.refinetrack_v01.build_data import ROOT,SOURCE,read_arrays,write_dataset,sha,dump
root=ROOT/'dataset/roban/refinetrack_v01/r2'
manifest=json.loads((root/'validation_manifest.json').read_text());meta=json.loads((SOURCE/'meta_motion.json').read_text());arrays=read_arrays(SOURCE);ids=[r['original_motion_id'] for r in manifest]
a={k:np.concatenate([v[meta['starts'][i]:meta['ends'][i]] for i in ids]) for k,v in arrays.items()}
clean=root/'validation/clean';output=root/'validation/legacy_clean'
write_dataset(output,a,json.loads((clean/'id_label.json').read_text()),json.loads((clean/'meta_motion.json').read_text()))
dump(root/'bridge_report.json',dict(status='GENERATED',generator_sha256=sha(__file__),files={str(f.relative_to(root)):sha(f) for f in output.rglob('*') if f.is_file()}))
