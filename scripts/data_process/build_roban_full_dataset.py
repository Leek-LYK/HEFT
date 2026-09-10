"""Audit the entire Ominisoma corpus, reject bad files, build bounded shards."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from active_adaptation.utils.roban_ominisoma import (
    load_roban_ominisoma_motion, sha256_file, resolve_confined_output_path,
)
from scripts.data_process.generate_dataset import check_motion
from scripts.data_process.generate_roban_dataset import CATEGORIES


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def quality_decision(motion, path):
    """Reject whole source files containing any interval rejected by the filter."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = check_motion(motion, None, path, 0, len(motion['qpos']))
    if result is False:
        reason = captured.getvalue().strip().split(': ', 1)[0]
        return False, [reason.removeprefix('Invalid motion due to ')], []
    intervals = result.get('bad_intervals', []) if isinstance(result, dict) else []
    return not intervals, sorted({row['reason'] for row in intervals}), intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-files', type=int, default=5000)
    parser.add_argument('--seed', default='20260904')
    args = parser.parse_args()
    if not 1 <= args.shard_files <= 5000:
        raise ValueError('shard-files must be in [1,5000]')
    source = args.source_root.resolve()
    output = resolve_confined_output_path(args.output, output_root=ROOT / 'dataset')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    started = time.monotonic()
    inventory = []
    for category in CATEGORIES:
        for path in (source / category).rglob('*.npz'):
            relative = path.relative_to(source).as_posix()
            digest = hashlib.sha256((args.seed + path.relative_to(source / category).as_posix()).encode()).hexdigest()
            inventory.append({'relative_path': relative, 'category': category, 'selection_hash': digest})
    inventory.sort(key=lambda row: (row['category'], row['selection_hash'], row['relative_path']))
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*.npz')}
    if actual != {r['relative_path'] for r in inventory}:
        raise ValueError('source contains NPZ outside the six supported categories')
    write_json(output / 'selection_manifest.json', {'sources': inventory, 'source_count': len(inventory),
               'selection': 'all six categories; within category sha256(seed + category-relative path)', 'seed': args.seed})
    accepted = {category: [] for category in CATEGORIES}
    rejected = Counter()
    reasons = Counter()
    with (output / 'source_manifest.jsonl').open('w') as manifest, (output / 'failed_files.jsonl').open('w') as failed:
        for index, row in enumerate(inventory, 1):
            path = source / row['relative_path']
            row['sha256_before'] = sha256_file(path)
            try:
                motion = load_roban_ominisoma_motion(path)
                row['source_frames'] = len(motion['qpos'])
                keep, why, intervals = quality_decision(motion, path)
                del motion
                row.update(accepted=keep, reasons=why, bad_intervals=intervals)
            except (ValueError, OSError, KeyError) as error:
                row.update(accepted=False, reasons=[type(error).__name__ + ': ' + str(error)], bad_intervals=[])
            if row['accepted']:
                accepted[row['category']].append(row)
            else:
                rejected[row['category']] += 1
                reasons.update(row['reasons'])
                failed.write(json.dumps(row) + '\n')
            manifest.write(json.dumps(row) + '\n')
            if index % 1000 == 0 or index == len(inventory):
                manifest.flush(); failed.flush()
                print(f'AUDIT {index}/{len(inventory)} accepted={sum(map(len, accepted.values()))} rejected={sum(rejected.values())}', flush=True)
    groups = []
    shard_reports = []
    for category in sorted(CATEGORIES):
        rows = accepted[category]
        for start in range(0, len(rows), args.shard_files):
            subset = rows[start:start + args.shard_files]
            name = f'{category}_shard_{start // args.shard_files:03d}'
            shard = output / name
            selected = output / 'logs' / (name + '_selection.json')
            write_json(selected, {'sources': subset})
            commands = [
                [sys.executable, '-m', 'scripts.data_process.generate_roban_dataset',
                 '--dataset-root', str(source), '--category', category, '--manifest', str(selected),
                 '--max-files', str(len(subset)), '--output', str(shard)],
                [sys.executable, '-m', 'scripts.data_process.validate_roban_shard',
                 '--source-root', str(source), '--mem-path', str(shard), '--motion-sample-count', '100',
                 '--output', str(shard / 'validation_report.json')],
            ]
            for stage, command in zip(('convert', 'validate'), commands):
                print(f'{stage.upper()} {name}: {len(subset)} sources', flush=True)
                with (output / 'logs' / f'{name}_{stage}.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            conversion = json.loads((shard / 'conversion_report.json').read_text())
            if conversion['successful_source_files'] != len(subset) or conversion['output_motion_count'] != len(subset):
                raise AssertionError(f'quality audit / conversion disagree for {name}')
            expected_hashes = {row['relative_path']: row['sha256_before'] for row in subset}
            for row in conversion['sources']:
                if expected_hashes[row['relative_path']] != row['sha256_before']:
                    raise AssertionError('source changed between quality audit and conversion')
            groups.append({'name': name, 'teacher_mem_path': str(shard.relative_to(ROOT / 'dataset')),
                           'weight': len(subset), 'sampling_mode': 'motion_uniform'})
            shard_reports.append({'name': name, 'category': category, 'source_files': len(subset),
                                  'motions': conversion['output_motion_count'], 'frames': conversion['output_total_frames']})
            write_json(output / 'dataset_groups.json', groups)
    # Verify every source, including rejected files, after the whole conversion.
    for index, row in enumerate(inventory, 1):
        if sha256_file(source / row['relative_path']) != row['sha256_before']:
            raise AssertionError('source changed: ' + row['relative_path'])
        if index % 10000 == 0:
            print(f'INTEGRITY {index}/{len(inventory)}', flush=True)
    from active_adaptation.utils.motion import GlobalMotionDataset
    import torch
    dataset = GlobalMotionDataset.load_from_mem_paths([g['teacher_mem_path'] for g in groups], ds_device=torch.device('cpu'))
    assert dataset.num_motions == sum(r['motions'] for r in shard_reports)
    assert dataset.num_steps == sum(r['frames'] for r in shard_reports)
    for name in ('root_pos_w', 'root_quat_w', 'joint_pos'):
        tensor = getattr(dataset.data, name)
        assert tensor.device.type == 'cpu' and tensor.dtype == torch.float32
        for chunk in tensor.split(100000):
            assert torch.isfinite(chunk).all()
    report = {'status': 'PASS', 'source_count': len(inventory), 'accepted_files': dataset.num_motions,
              'rejected_files': sum(rejected.values()), 'motions': dataset.num_motions, 'frames': dataset.num_steps,
              'shards': shard_reports, 'categories': {c: {'source': sum(r['category'] == c for r in inventory),
              'accepted': len(accepted[c]), 'rejected': rejected[c]} for c in CATEGORIES},
              'rejection_reasons': dict(reasons), 'source_hashes_unchanged': True,
              'quality_policy': 'whole-file rejection using existing check_motion; any bad interval rejects file',
              'storage_dtype': 'float16', 'loader_dtype': 'float32', 'dataset_device': 'cpu',
              'all_shards_validated': True, 'joint_order': dataset.joint_names,
              'dataset_tensor_bytes_cpu': sum(getattr(dataset.data,k).numel()*4 for k in ('root_pos_w','root_quat_w','joint_pos')),
              'elapsed_seconds': time.monotonic() - started,
              'data_and_audit_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
    write_json(output / 'full_report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('shards', 'joint_order')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
