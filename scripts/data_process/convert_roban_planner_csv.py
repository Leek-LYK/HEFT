"""Convert named 25 Hz Roban planner CSVs into the HEFT 50 Hz reference contract.

The source root convention must be explicit. This converter does not retarget
human motion, infer locomotion translation, repair contacts, or add entry blends.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

JOINT_NAMES = (
    'waist_yaw_joint',
    *(f'leg_{side}{i}_joint' for side in ('l', 'r') for i in range(1, 7)),
    *(f'zarm_{side}{i}_joint' for side in ('l', 'r') for i in range(1, 5)),
)
INPUT_FIELDS = (
    'timestamp', 'root_translateX', 'root_translateY', 'root_translateZ',
    'root_quat_w', 'root_quat_x', 'root_quat_y', 'root_quat_z',
    *(name + '_dof' for name in JOINT_NAMES),
)
OUTPUT_FIELDS = (
    'body_pos_x', 'body_pos_y', 'body_pos_z',
    'body_quat_w', 'body_quat_x', 'body_quat_y', 'body_quat_z',
    *(f'joint_pos{i:02d}' for i in range(21)),
    *(f'joint_vel_{i:02d}' for i in range(21)),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_motion(source: Path, root_frame: str) -> tuple[np.ndarray, dict]:
    if root_frame not in ('chest', 'pelvis'):
        raise ValueError('root_frame must be chest or pelvis')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f'duplicate columns: {source}')
        if set(fields) != set(INPUT_FIELDS):
            raise ValueError(f'{source}: missing={set(INPUT_FIELDS)-set(fields)}, extra={set(fields)-set(INPUT_FIELDS)}')
        raw = np.array([[float(row[k]) for k in INPUT_FIELDS] for row in reader], dtype=np.float64)
    if raw.ndim != 2 or len(raw) < 3 or not np.isfinite(raw).all():
        raise ValueError(f'{source}: requires at least three finite frames')
    times = raw[:, 0] - raw[0, 0]
    if not np.allclose(np.diff(times), 1/25, rtol=0, atol=1e-5):
        raise ValueError(f'{source}: timestamps must be strictly increasing at 25 Hz')
    # Preserve the first/last sample times: 100 frames -> 199 at 50 Hz.
    target_times = np.arange(2 * len(raw) - 1, dtype=np.float64) / 50
    if not np.isclose(target_times[-1], times[-1], atol=1e-5):
        raise ValueError('source duration does not match target grid')
    # Avoid tiny source timestamp rounding differences at the last Slerp query.
    source_times = np.arange(len(raw), dtype=np.float64) / 25
    pos = np.column_stack([np.interp(target_times, source_times, raw[:, j]) for j in range(1, 4)])
    joints = np.column_stack([np.interp(target_times, source_times, raw[:, j]) for j in range(8, 29)])
    quats = raw[:, 4:8].copy()
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError(f'{source}: zero quaternion')
    quats /= norms[:, None]
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] *= -1
    rotations = Slerp(source_times, Rotation.from_quat(np.roll(quats, -1, axis=1)))(target_times)
    if root_frame == 'pelvis':
        # Supplied official S17 XML: torso translation=0, waist axis=+Z.
        # Chest pose is pelvis pose composed with old hardware waist rotation.
        rotations = rotations * Rotation.from_euler('z', joints[:, 0])
        joints[:, 0] *= -1
    out_quats = np.roll(rotations.as_quat(), 1, axis=1)
    for i in range(1, len(out_quats)):
        if np.dot(out_quats[i-1], out_quats[i]) < 0:
            out_quats[i] *= -1
    velocities = np.gradient(joints, 1/50, axis=0, edge_order=2)
    output = np.column_stack([pos, out_quats, joints, velocities])
    if output.shape[1] != 49 or not np.isfinite(output).all():
        raise ValueError('invalid output')
    report = {
        'source': str(source.resolve()), 'source_sha256': sha256(source),
        'source_root_frame': root_frame, 'output_root_frame': 'chest',
        'source_hz': 25, 'output_hz': 50,
        'source_frames': len(raw), 'output_frames': len(output),
        'timestamp_span_s': float(target_times[-1]),
        'nominal_playback_s': len(output)/50,
        'interpolation': {'translation': 'linear', 'joints': 'linear, no unwrap or clipping', 'orientation': 'normalized shortest-path SLERP', 'joint_velocity': 'finite differences on output policy-coordinate joint angles'},
        'root_translation_range_m': np.ptp(pos, axis=0).tolist(),
        'root_translation_is_constant': bool(np.all(np.ptp(pos, axis=0) < 1e-10)),
        'quaternion_norm_error_before_normalization': float(np.max(np.abs(norms-1))),
        'max_abs_joint_velocity_rad_s': float(np.max(np.abs(velocities))),
        'first_joint_positions_rad': joints[0].tolist(),
        'last_joint_positions_rad': joints[-1].tolist(),
        'max_abs_joint_velocity_by_name': dict(zip(JOINT_NAMES, np.max(np.abs(velocities), axis=0).tolist())),
    }
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root-frame', choices=('chest', 'pelvis'), required=True,
                        help='chest: latest training XML coordinates; pelvis: supplied old official XML/hardware waist sign')
    args = parser.parse_args()
    files = sorted(args.input_dir.glob('*.csv'))
    if not files:
        raise ValueError('no CSV files found')
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite output directory: {args.output_dir}')
    # Validate/convert all sources before writing any output.
    converted = [(source, *convert_motion(source, args.source_root_frame)) for source in files]
    args.output_dir.mkdir(parents=True)
    reports=[]
    for source, data, report in converted:
        dest=args.output_dir/(source.stem+'_heft_50hz.csv')
        np.savetxt(dest, data, delimiter='\t', header='\t'.join(OUTPUT_FIELDS), comments='', fmt='%.10f')
        reloaded=np.loadtxt(dest, skiprows=1)
        np.testing.assert_allclose(reloaded, data, atol=5.1e-11, rtol=0)
        assert sha256(source)==report['source_sha256']
        report.update(output=str(dest.resolve()), output_sha256=sha256(dest), finite=True)
        reports.append(report)
    manifest={'format':'HEFT CSV: body_pos(3), body_quat wxyz(4), policy joint_pos(21), joint_vel(21)',
              'joint_names':JOINT_NAMES, 'motions':reports,
              'limitations':['Only format, coordinate convention and sampling conversion; not a sim2real performance validation.',
                             'No root translation is reconstructed; constant planner positions remain constant.',
                             'No stand-to-motion transition, contact repair or joint-limit clipping is added.']}
    (args.output_dir/'conversion_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with (args.output_dir/'motion_entries.yaml').open('w') as f:
        f.write('# Merge entries into HumanoidRobotCfg.motion.motions after copying CSVs to motion_data.\n')
        for source,_,_ in converted:
            name=source.stem
            f.write(f'- name: {name}\n  file: "../motion_data/{name}_heft_50hz.csv"\n')
    print(json.dumps({'output_dir':str(args.output_dir),'motions':len(reports),'frames':[r['output_frames'] for r in reports]},indent=2))


if __name__=='__main__':
    main()
