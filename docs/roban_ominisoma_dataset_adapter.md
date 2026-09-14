# Roban Ominisoma dataset adapter

This document defines the conversion boundary between the legacy
Ominisoma/IsaacLab Roban export and HEFT's current Roban asset. It does not
change the G1/L7 motion contract.

## Existing HEFT pipeline

The standard input accepted by `scripts/data_process/generate_dataset.py` is a
50 Hz NPZ with these fields:

| Field | Contract |
| --- | --- |
| `root_pos` | `[T, 3]`, world position |
| `root_rot` | `[T, 4]`, quaternion in **xyzw** order |
| `dof_pos` | `[T, J]` |
| `local_body_pos` | `[T, B, 3]`, positions expressed in the rotated root frame |
| `joint_names` | ordered names for the first `J` DOF columns |
| `body_names` | ordered names for the `B` body columns |
| `mocap_framerate`, `frequency`, or `fps` | must resolve to 50 |

The standard loader converts `root_rot` exactly once from xyzw to wxyz and
forms internal MuJoCo qpos as

```text
[root_xyz, root_quaternion_wxyz, joint_position]
```

It reconstructs source world positions for filtering as

```text
body_pos_w = R_world_root @ local_body_pos + root_pos_w
```

The writer stores only `root_pos_w`, `root_quat_w` (wxyz), and `joint_pos` in
`_tensordict`. `meta_motion.json` contains `joint_names`, legacy `body_names`,
motion `starts`/`ends`, and optional per-motion information. `id_label.json`
contains source/segment labels. Root/body velocities and body poses are not
stored.

At runtime, `ProgressiveMultiMotionDataset` selects body names from the current
asset and `MotionFKHelper` reconstructs all body poses and finite-difference
velocities from the three minimal tensors. Stored source body poses are never
used as runtime targets. Multiple configured dataset paths are accepted and
combined, but the current loader converts every shard to float32 and then uses
`torch.cat`. With `keep_on_cpu: true`, `MotionTrackingCommand` keeps the full
reference dataset on CPU regardless of environment count; only bounded runtime
windows are transferred to the simulation device. Without this option, storage
is selected on CPU below 1024 envs and on the simulation device at 1024 envs or
above. The successful 50K Roban training uses `keep_on_cpu: true`.

The existing writer likewise retains one tensor chunk per accepted motion and
performs a final `torch.cat`; an unbounded conversion is therefore unsuitable
for the full Roban corpus.

## Ominisoma source schema

Every audited source file uses:

| Field | Shape | Dtype |
| --- | --- | --- |
| `body_ang_vel_w` | `[T, 28, 3]` | `float32` |
| `body_lin_vel_w` | `[T, 28, 3]` | `float32` |
| `body_pos_w` | `[T, 28, 3]` | `float32` |
| `body_quat_w` | `[T, 28, 4]` | `float32`, **wxyz** |
| `joint_pos` | `[T, 21]` | `float32` |
| `joint_vel` | `[T, 21]` | `float32` |
| `fps` | `[1]` | `int64`, value 50 |

The root candidate is `base_link`, body index 0. The source does not carry
joint/body names, so the following fixed contracts are part of the adapter.

### Joint order

Source order:

```text
waist_yaw,
zarm_l1, zarm_r1, leg_l1, leg_r1,
zarm_l2, zarm_r2, leg_l2, leg_r2,
zarm_l3, zarm_r3, leg_l3, leg_r3,
zarm_l4, zarm_r4, leg_l4, leg_r4,
leg_l5, leg_r5, leg_l6, leg_r6
```

Target order is imported directly from `ROBAN_JOINT_ORDER`:

```text
waist_yaw_joint,
leg_l1_joint, leg_l2_joint, leg_l3_joint, leg_l4_joint, leg_l5_joint, leg_l6_joint,
leg_r1_joint, leg_r2_joint, leg_r3_joint, leg_r4_joint, leg_r5_joint, leg_r6_joint,
zarm_l1_joint, zarm_l2_joint, zarm_l3_joint, zarm_l4_joint,
zarm_r1_joint, zarm_r2_joint, zarm_r3_joint, zarm_r4_joint
```

The name-derived source-to-target indices are:

```text
(0, 3, 7, 11, 15, 17, 19, 4, 8, 12, 16, 18, 20,
 1, 5, 9, 13, 2, 6, 10, 14)
```

Both position and velocity arrays use this permutation. Missing, duplicate, or
unknown contract names are errors. Waist yaw is not negated.

### Body order

```text
base_link, waist_yaw_link, zarm_l1_link, zarm_r1_link, zhead_1_link,
leg_l1_link, leg_r1_link, zarm_l2_link, zarm_r2_link, zhead_2_link,
head_radar, leg_l2_link, leg_r2_link, zarm_l3_link, zarm_r3_link,
camera_base, leg_l3_link, leg_r3_link, zarm_l4_link, zarm_r4_link,
leg_l4_link, leg_r4_link, zarm_l5_link, zarm_r5_link,
leg_l5_link, leg_r5_link, leg_l6_link, leg_r6_link
```

`zarm_l5_link`, `zarm_r5_link`, and `camera_base` belong to the legacy asset.
They are not renamed to `l_handball`, `r_handball`, or `head_camera_base`.
Source world body states are used only for quality/FK audits and preprocessing
filters. The Roban adapter writes an empty legacy `body_names` metadata list;
runtime body states and names come exclusively from the current MJCF FK.

The configured source feet are `leg_l6_link` and `leg_r6_link`. The generic
preprocessor now accepts `foot_names`; its default remains
`left_ankle_roll_link` and `right_ankle_roll_link` for G1/L7.

## Quaternion and local-position boundary

Source quaternions are always interpreted as wxyz. The adapter checks finite
values and nonzero norms, rejects a maximum norm deviation above `1e-2`,
normalizes accepted values, counts adjacent negative dot products, and applies
the physically equivalent `q/-q` sign correction along time. It never probes a
second ordering.

If a standard-format xyzw value is needed, conversion is exactly:

```text
root_rot_xyzw = root_quat_wxyz[..., [1, 2, 3, 0]]
```

The inverse conversion is tested element-by-element. The adapter can consume
world positions directly, so it does not manufacture a second standard NPZ.
For callers that need local positions, the correct equations are:

```text
local_body_pos = R_world_root.T @ (body_pos_w - root_pos_w)
body_pos_w = root_pos_w + R_world_root @ local_body_pos
```

Both directions and a non-identity rotated-root round trip are covered by tests.

## FK audit result

The audit deterministically used two files from each of `acrobatic`, `crawl`,
`jump`, `kneel`, `main`, and `squat`. Those 12 files contained 10,824 frames;
16 evenly spaced frames per file (192 frames total) were compared against
`biped_s17_verified_handball_fixedhead.xml`.

- All audited arrays had the schema and dtypes above, with no NaN/Inf.
- Maximum quaternion norm deviation was `2.77e-7`.
- There were 31,920 source body quaternion sign flips, including 1,140 root
  flips. These are removed only by the equivalent q/-q continuity operation.
- `base_link` position and orientation reproduced exactly for both waist trials.
- With `waist_sign=+1`, common-body position mean/p95/max was
  `0.00115063 / 0.0000017804 / 0.0264540 m`; orientation mean/p95/max was
  `4.38e-7 / 9.06e-7 / 1.42e-6 rad`.
- The aggregate position mean is dominated by a fixed `zhead_2_link`
  translation difference: its mean is `0.0264530 m`, while every other common
  body has mean error below `0.7e-6 m` and max below `2.45e-6 m`. Its orientation
  still agrees within `2.69e-7 rad`. This is an asset-version frame translation,
  not a root, leg, arm, or waist-sign mismatch.
- With `waist_sign=-1`, common-body position mean/p95/max worsened to
  `0.0176115 / 0.0776186 / 0.509065 m`; orientation mean/p95/max worsened to
  `0.113229 / 0.531694 / 1.73588 rad`.

The evidence therefore supports `base_link` index 0, wxyz, and
`waist_sign=+1`. The isolated legacy/current fixed-head translation is not
copied into the runtime dataset.

## Bounded conversion CLI

The converter requires an explicit bound and refuses existing output unless
`--overwrite` is supplied. It sorts inputs deterministically, validates each
file independently, records failures, hashes sources before and after
conversion, writes to a sibling temporary directory, and publishes only after
the memdataset and reports are complete.

Example single-file conversion (the manifest contains a relative NPZ path):

```bash
uv run --no-sync python scripts/data_process/generate_roban_dataset.py \
  --dataset-root "$SOURCE_ROOT" \
  --category main \
  --manifest /tmp/roban_single_manifest.txt \
  --max-files 1 \
  --max-total-frames 1500 \
  --output "$MEMPATH/roban/main_one"
```

Example bounded shard:

```bash
uv run --no-sync python scripts/data_process/generate_roban_dataset.py \
  --dataset-root "$SOURCE_ROOT" \
  --category main \
  --start-index 0 \
  --max-files 3000 \
  --max-total-frames 2000000 \
  --output "$MEMPATH/roban/main_shard_0000"
```

Use `next_start_index` from `conversion_report.json` for the next
`--start-index`; a file that would exceed the frame cap is deliberately left for
the next shard. Each completed output includes a relative-path source manifest,
before/after hashes, its manifest digest, counts, and failures. An interrupted
run leaves only a hidden temporary directory, which is removed on handled
errors; it never publishes a partial output under the requested name.

## Current training dataset and memory limits

The successful training baseline uses `roban/main/train_50000_v1` with
`keep_on_cpu: true`. See the [training guide](使用说明.md) for commands and the
[success record](roban_training_sim2real_success_20260908.md) for the exact run.
The dataset's selection manifest and conversion report define the selected
source files and valid segments; 50K selected files does not imply 50K segments.

CPU placement avoids storing the full reference corpus on each GPU. It does
not make loading lazy: each DDP rank still loads float32 tensors and concatenates
them independently, so conversion and host memory budgets must remain bounded.
Multiple shards must be listed explicitly; the loader does not recursively
discover a shard directory. Select dataset group weights deliberately.

The small `_smoke/roban/main_subset` dataset remains an input to the smoke and
pilot configurations. `roban/main/shard_000` remains the Teacher evaluation
script's default dataset; neither is an obsolete cache.

## Historical validation

The initial single-file and 50-file smoke results, full-corpus estimates and
then-pending work are preserved in the
[data-stage historical snapshot](archive/roban_ominisoma_dataset_adapter_pre_cleanup_20260909.md).
Those pending-work statements describe that earlier stage, not current support.
Data conversion and FK checks alone do not establish policy or deployment quality;
consult the success record for the actual validation scope.
