"""Compile paired XWBC data without changing source files."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from active_adaptation.utils.xwbc_data import build_dataset, sha256

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit-per-split', type=int)
    p.add_argument('--shard-size', type=int, default=128)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if a.verify_only:
        root = Path(a.output)
        schema = json.loads((root/'dataset.json').read_text())
        if (root/'READY').read_text().strip() != sha256(root/'dataset.json'):
            raise ValueError('Dataset completion digest mismatch')
        if schema['manifest_sha256'] != sha256(a.manifest):
            raise ValueError('Manifest changed; choose a new DATASET directory')
        print(f'Verified prepared dataset: {root}')
    else:
        build_dataset(a.manifest, a.output, a.workers, a.limit_per_split, a.shard_size)
