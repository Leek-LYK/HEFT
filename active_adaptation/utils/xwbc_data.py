"""Paired motion compiler and command bank for XWBC (50 Hz, SI, wxyz)."""
from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('human', 'g1', 'roban')
ROBOTS = ('g1', 'roban')
XMLS = {'g1': 'active_adaptation/assets/G1/g1.xml',
        'roban': 'active_adaptation/assets/roban/mjcf/biped_s17_verified_handball_fixedhead.xml'}
BODIES = {
    'g1': ('pelvis', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_mimic',
           'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_mimic',
           'left_hip_yaw_link', 'left_knee_link', 'left_ankle_roll_link',
           'right_hip_yaw_link', 'right_knee_link', 'right_ankle_roll_link'),
    'roban': ('waist_yaw_link', 'zarm_l3_link', 'zarm_l4_link', 'l_handball',
              'zarm_r3_link', 'zarm_r4_link', 'r_handball', 'leg_l1_link',
              'leg_l4_link', 'leg_l6_link', 'leg_r1_link', 'leg_r4_link', 'leg_r6_link'),
}


@lru_cache(maxsize=None)
def literal(path, name):
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def joint_names(robot):
    return literal(f'active_adaptation/assets/{robot}.py', f'{robot.upper()}_JOINT_ORDER')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(rows):
    ids, families = set(), {}
    for row in rows:
        if row['motion_id'] in ids:
            raise ValueError(f"duplicate motion_id: {row['motion_id']}")
        ids.add(row['motion_id'])
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('unknown split')
        family = row['family_id']
        if family in families and families[family] != row['split']:
            raise ValueError(f'family leakage: {family}')
        families[family] = row['split']
        if set(row['paths']) != set(SOURCES):
            raise ValueError('every motion needs all three sources')


def heading_features(pos, rot, velocity):
    """Root-centered, heading-relative body features + world root height and yaw rate.

    Root here is the semantic pelvis (body 0), independently of the floating base.
    Per-source encoders handle the different skeletal axes; no target state enters.
    """
    yaw = np.unwrap(np.arctan2(rot[:, 0, 1, 0], rot[:, 0, 0, 0]))
    inv = Rotation.from_euler('z', -yaw).as_matrix()
    local_pos = np.einsum('tij,tbj->tbi', inv, pos - pos[:, :1])
    local_rot = np.einsum('tij,tbjk->tbik', inv, rot)
    local_vel = np.einsum('tij,tbj->tbi', inv, velocity)
    rot6 = np.concatenate((local_rot[..., :, 0], local_rot[..., :, 1]), -1)
    body = np.concatenate((local_pos, rot6, local_vel), -1).reshape(len(pos), -1)
    # Preserve intended turning and translation, even after heading normalization.
    extra = np.column_stack((pos[:, 0, 2], np.gradient(yaw, .02)))
    return np.concatenate((body, extra), -1).astype(np.float32)


_MODELS = {}


def load_robot(row, robot):
    names = joint_names(robot)
    path = ROOT / row['paths'][robot]
    if robot == 'g1':
        with path.open() as f:
            columns = f.readline().strip().split(',')
        values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        if len(columns) != 36 or not np.array_equal(values[:, 0], np.arange(len(values))):
            raise ValueError('G1 CSV frame index/schema mismatch')
        pos = values[:, [columns.index('root_translate' + a) for a in 'XYZ']] / 100
        euler = values[:, [columns.index('root_rotate' + a) for a in 'XYZ']]
        quat = Rotation.from_euler('xyz', euler, degrees=True).as_quat()[:, [3, 0, 1, 2]]
        joints = np.deg2rad(values[:, [columns.index(n + '_dof') for n in names]])
    else:
        source_names = literal('active_adaptation/utils/roban_ominisoma.py', 'SOURCE_JOINT_NAMES')
        with np.load(path, allow_pickle=False) as z:
            if int(z['fps'].reshape(-1)[0]) != 50:
                raise ValueError('Roban fps must be 50')
            pos, quat = z['body_pos_w'][:, 0].copy(), z['body_quat_w'][:, 0].copy()
            joints = z['joint_pos'][:, [source_names.index(n.removesuffix('_joint')) for n in names]]
    if not all(np.isfinite(a).all() for a in (pos, quat, joints)):
        raise ValueError(f'{robot}: nonfinite reference')
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(np.abs(norms - 1) > .01):
        raise ValueError(f'{robot}: quaternion norm')
    quat = quat / norms
    if robot not in _MODELS:
        m = mujoco.MjModel.from_xml_path(str(ROOT / XMLS[robot]))
        _MODELS[robot] = m, mujoco.MjData(m)
    model, data = _MODELS[robot]
    indices = [model.body(n).id for n in BODIES[robot]]
    qindices = [int(model.joint(n).qposadr[0]) for n in names]
    bp = np.empty((len(pos), len(indices), 3))
    br = np.empty((len(pos), len(indices), 3, 3))
    for t in range(len(pos)):
        data.qpos[:] = model.qpos0
        data.qpos[:3], data.qpos[3:7], data.qpos[qindices] = pos[t], quat[t], joints[t]
        mujoco.mj_kinematics(model, data)
        bp[t], br[t] = data.xpos[indices], data.xmat[indices].reshape(-1, 3, 3)
    command = heading_features(bp, br, np.gradient(bp, .02, axis=0))
    # Record gross geometric issues. Keep common eligibility explicit and conservative.
    diagnostics = dict(min_body_z=float(bp[..., 2].min()),
                       max_joint_speed=float(np.abs(np.gradient(joints, .02, axis=0)).max()))
    return dict(root_pos=pos.astype(np.float32), root_quat=quat.astype(np.float32),
                joints=joints.astype(np.float32), command=command, diagnostics=diagnostics)


def compile_motion(row):
    try:
        with np.load(ROOT / row['paths']['human'], allow_pickle=False) as z:
            h = z['motion'].copy()
            if int(z['fps']) != 50 or h.shape[1:] != (22, 12):
                raise ValueError('human schema/fps')
        if len(h) < 16 or not np.isfinite(h).all():
            raise ValueError('human finite/minimum length')
        a, b = h[..., 3:6], h[..., 6:9]
        if max(np.abs(np.linalg.norm(a, axis=-1)-1).max(), np.abs(np.linalg.norm(b, axis=-1)-1).max(),
               np.abs((a*b).sum(-1)).max()) > .01:
            raise ValueError('human rotation columns')
        hr = np.stack((a, b, np.cross(a, b)), -1)
        result = {r: load_robot(row, r) for r in ROBOTS}
        if any(len(v['joints']) != len(h) for v in result.values()):
            raise ValueError('paired frame count mismatch')
        command = heading_features(h[..., :3], hr, h[..., 9:12])
        arrays = [command, *(result[r]['command'] for r in ROBOTS)]
        if any(not np.isfinite(a).all() or np.abs(a).max() > 60000 for a in arrays):
            raise ValueError('command numeric range')
        return dict(row=row, human=command, robots=result, frames=len(h),
                    source_hashes={s: sha256(ROOT / p) for s, p in row['paths'].items()})
    except (ValueError, KeyError, OSError) as e:
        return dict(row=row, error=f'{type(e).__name__}: {e}')


def build_dataset(manifest, output, workers=4, limit_per_split=None, shard_size=128):
    """Stream bounded shards. Output must be new; READY is written only on success."""
    import torch
    from active_adaptation.utils.motion import MotionMinimalData

    if workers < 1 or shard_size < 1 or (limit_per_split is not None and limit_per_split < 1):
        raise ValueError("workers, shard_size and optional limit must be positive")
    output = Path(output).resolve()
    rows = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    validate_manifest(rows)
    if limit_per_split:
        limited = []
        for split in ('train', 'val', 'test'):
            # Round-robin categories, retaining reproducible manifest ordering.
            cats = {c: [r for r in rows if r['split'] == split and r['category'] == c]
                    for c in sorted({r['category'] for r in rows})}
            ordered = [v[i] for i in range(max(map(len, cats.values()))) for v in cats.values() if i < len(v)]
            limited.extend(ordered[:limit_per_split])
        rows = limited
    output.mkdir(parents=True, exist_ok=False)
    (output / 'input_manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    schema = dict(version=1, fps=50, sources=list(SOURCES), robots=list(ROBOTS),
                  offsets=[-4, -2, 0, 2, 4], manifest_sha256=sha256(manifest),
                  asset_hashes={r: sha256(ROOT / XMLS[r]) for r in ROBOTS},
                  command_features='pelvis heading-relative xyz/rotation columns/world velocity; pelvis z/yaw_rate',
                  splits={}, rejected=[], joint_names={r: list(joint_names(r)) for r in ROBOTS})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split in ('train', 'val', 'test'):
            split_rows = [r for r in rows if r['split'] == split]
            shards = []
            for offset in range(0, len(split_rows), shard_size):
                # At most one shard is in flight; tensor memory stays bounded.
                results = list(pool.map(compile_motion, split_rows[offset:offset+shard_size]))
                schema['rejected'].extend({**x['row'], 'reason': x['error']} for x in results if 'error' in x)
                results = [x for x in results if 'error' not in x]
                if not results:
                    continue
                name = f'{split}/shard_{len(shards):04d}'
                folder = output / name
                folder.mkdir(parents=True)
                lengths = [x['frames'] for x in results]
                ends = np.cumsum(lengths).tolist(); starts = [0] + ends[:-1]
                labels = [{**x['row'], 'frames': x['frames'], 'source_hashes': x['source_hashes'],
                           'diagnostics': {r: x['robots'][r]['diagnostics'] for r in ROBOTS}}
                          for x in results]
                (folder / 'motions.json').write_text(json.dumps(labels, ensure_ascii=False))
                for source in SOURCES:
                    values = np.concatenate([x['human'] if source == 'human' else x['robots'][source]['command'] for x in results])
                    np.save(folder / f'{source}_command.npy', values.astype(np.float16))
                for robot in ROBOTS:
                    target = folder / robot
                    target.mkdir()
                    arrays = {k: torch.from_numpy(np.concatenate([x['robots'][robot][k] for x in results]))
                              for k in ('root_pos', 'root_quat', 'joints')}
                    data = MotionMinimalData(root_pos_w=arrays['root_pos'], root_quat_w=arrays['root_quat'],
                                             joint_pos=arrays['joints'], batch_size=[ends[-1]])
                    data.memmap(str(target / '_tensordict'))
                    (target / 'meta_motion.json').write_text(json.dumps(dict(joint_names=list(joint_names(robot)),
                                                                          body_names=[], starts=starts, ends=ends, info={})))
                    (target / 'id_label.json').write_text(json.dumps([dict(source_path=x['row']['motion_id'],
                        segment_start=0, segment_end=x['frames']) for x in results]))
                shards.append(dict(path=name, count=len(results), frames=ends[-1], starts=starts, ends=ends))
                print(f'{name}: motions={len(results)} frames={ends[-1]}', flush=True)
            if not shards:
                raise ValueError(f'no valid motions in {split}')
            schema['splits'][split] = shards
    (output / 'dataset.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    (output / 'READY').write_text(sha256(output / 'dataset.json') + '\n')
    return schema


class CommandBank:
    """Same motion ordering as HEFT shard groups; banks stay in fp16 on device."""
    def __init__(self, root, split, device):
        import torch
        self.root = Path(root).resolve()
        self.schema = json.loads((self.root / 'dataset.json').read_text())
        if (self.root / 'READY').read_text().strip() != sha256(self.root / 'dataset.json'):
            raise ValueError('dataset completion digest mismatch')
        for robot, expected in self.schema.get("asset_hashes", {}).items():
            if sha256(ROOT / XMLS[robot]) != expected:
                raise ValueError(f"{robot} asset changed; rebuild the paired reference dataset")
        self.split, self.device = split, device
        self.shards = self.schema['splits'][split]
        self.rows, starts, ends, cursor = [], [], [], 0
        for shard in self.shards:
            self.rows.extend(json.loads((self.root / shard['path'] / 'motions.json').read_text()))
            starts.extend(cursor + n for n in shard['starts']); ends.extend(cursor + n for n in shard['ends'])
            cursor += shard['frames']
        self.starts = torch.tensor(starts, device=device)
        self.ends = torch.tensor(ends, device=device)
        self.lengths = self.ends - self.starts
        self.offsets = torch.tensor(self.schema['offsets'], device=device)
        self.banks = {}
        for source in SOURCES:
            parts = [torch.from_numpy(np.load(self.root / sh['path'] / f'{source}_command.npy')) for sh in self.shards]
            self.banks[source] = torch.cat(parts).to(device)
        self.dims = {s: a.shape[-1] * len(self.offsets) + len(self.offsets) for s, a in self.banks.items()}

    def groups(self, robot):
        return [dict(name=f'xwbc_{i}', teacher_mem_path=str(self.root / sh['path'] / robot), weight=sh['count'])
                for i, sh in enumerate(self.shards)]

    def gather(self, motion, frame):
        import torch
        t = frame.long()[:, None] + self.offsets
        length = self.lengths[motion][:, None]
        valid = (t >= 0) & (t < length)
        indices = self.starts[motion][:, None] + torch.minimum(t.clamp_min(0), length-1)
        return {s: torch.cat((a[indices].float().flatten(1), valid.float()), -1) for s, a in self.banks.items()}
