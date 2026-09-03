# Roban S22 / Leju HEFT first-version training contract

## Scope and asset

This is the first training-side adapter for the Roban S22 handball, fixed-head
variant. It intentionally does not add motion data, start PPO training, export a
deployment ONNX, or perform Sim2Real.

The authoritative training MJCF is:

`active_adaptation/assets/roban/mjcf/biped_s17_verified_handball_fixedhead.xml`

It comes from the local `roban_s22_0901.zip` delivery. The adapter copies the
MJCF, its 28 referenced meshes, and `package.xml` solely to retain delivered
version/license metadata. That metadata declares version `1.0.0` and BSD
licensing. The URDF, scene XML and unrelated meshes are not needed by training
and are omitted.

Naming note: the delivered archive/directory calls the asset
`roban_s22_handball`, while the authoritative filename/MJCF model and Leju live
configuration retain `biped_s17` / `roban_s17` identifiers. The adapter keeps
those source identifiers unchanged and registers the training robot as `roban`.

Both `mujoco.MjModel.from_xml_path()` and `MjSpec.compile()` produce `nq=28`,
`nv=27`, `nu=21`, `njnt=22`, `nbody=38`, and `ngeom=65`. Summing compiled
`body_mass` gives `40.69991336 kg`. This is authoritative for this adapter and
differs from the older `40.186 kg` value in the delivered README.

The source geoms are unnamed. The adapter gives them deterministic in-memory
`<body>_visual_<n>` or `<body>_collision_<n>` names so collision configuration
can be validated without changing any physical property. Source torque motors
are also removed in memory and replaced by training position actuators; the
source MJCF itself remains unchanged.

## Canonical coordinates and bodies

The floating root is `base_link`. Its verified initial world position is
`[0, 0, 0.77]`. The policy controls 21 hinge joints and does not control the free
joint or the fixed head.

Canonical policy joint order:

| Index | Joint |
|---:|---|
| 0 | `waist_yaw_joint` |
| 1 | `leg_l1_joint` |
| 2 | `leg_l2_joint` |
| 3 | `leg_l3_joint` |
| 4 | `leg_l4_joint` |
| 5 | `leg_l5_joint` |
| 6 | `leg_l6_joint` |
| 7 | `leg_r1_joint` |
| 8 | `leg_r2_joint` |
| 9 | `leg_r3_joint` |
| 10 | `leg_r4_joint` |
| 11 | `leg_r5_joint` |
| 12 | `leg_r6_joint` |
| 13 | `zarm_l1_joint` |
| 14 | `zarm_l2_joint` |
| 15 | `zarm_l3_joint` |
| 16 | `zarm_l4_joint` |
| 17 | `zarm_r1_joint` |
| 18 | `zarm_r2_joint` |
| 19 | `zarm_r3_joint` |
| 20 | `zarm_r4_joint` |

Three orders are kept explicit:

- MJCF hinge tree: waist, left leg, right leg, left arm, right arm. It currently
  happens to equal canonical policy order.
- Source MJCF actuator order: left leg, right leg, waist, left arm, right arm.
- Training position actuator order: explicitly sorted into canonical policy
  order.

Body semantics:

| Role | Body |
|---|---|
| Floating root / IMU chest | `base_link` |
| Pelvis / waist side | `waist_yaw_link` |
| Head | `zhead_2_link` |
| Hands | `l_handball`, `r_handball` |
| Feet | `leg_l6_link`, `leg_r6_link` |

`torso_pattern` is `^waist_yaw_link$`. In the common config this field is used
for COM and wrench randomization, while `base_pattern` is already included in
COM randomization. Reusing `base_link` for both fields would make the resolver
see a duplicate match. Selecting `waist_yaw_link` covers the pelvis-side body
and keeps the floating root unambiguously `base_link`.

`waist_yaw_joint` uses `direction=-1` only at the canonical-policy ↔ hardware
boundary in the Leju controller. The training MJCF axis, joint order, actions,
observations and motion coordinates do not apply that sign a second time.

Symmetry validation is exact for the 26 physical/link bodies used by training:
structure and sampled-FK position/orientation errors are all `0` at a `1e-6`
tolerance. An intentionally unfiltered check over every XML body reports
structure errors `0.05 m / 2.0` (rotation-matrix max norm) and sampled-FK errors
`0.0500000007 m / 1.9267287`; these come from camera optical coordinate frames
whose names/poses are not sagittal physical links. They are excluded from Roban's
validation body set instead of weakening the global tolerance.

## Actuation and actions

All actuated MJCF joints have verified armature `0.003`. Position-control
parameters from the successful Leju deployment contract are:

| Joint group | Kp | Kd | Effort limit |
|---|---:|---:|---:|
| `waist_yaw_joint` | 40.1792 | 3.5579 | 80.0 |
| `leg_[lr]1_joint` | 90.1792 | 3.5579 | 150.0 |
| `leg_[lr]2_joint` | 150.0984 | 8.3088 | 150.0 |
| `leg_[lr]3_joint` | 40.1792 | 3.5579 | 70.0 |
| `leg_[lr]4_joint` | 150.0984 | 8.3088 | 150.0 |
| `leg_[lr]5_joint` | 34.2506 | 2.9072 | 74.0 |
| `leg_[lr]6_joint` | 34.2506 | 2.9072 | 74.0 |
| `zarm_[lr]1_joint` | 14.2506 | 1.9072 | 14.1 |
| `zarm_[lr][2-4]_joint` | 14.2506 | 1.9072 | 37.0 |

The first-version `soft_joint_pos_limit_factor=0.9` follows the existing
framework convention and is provisional.

Action scale is `0.5` for waist and legs and `1.0` for arms. Deployment uses
absolute action targets (`residual_action=false`), zero default joint position,
policy clipping to `[-10, 10]`, and filtering
`applied = 0.1 * previous + 0.9 * new`. Training retains the existing action
alpha domain randomization `[0.8, 1.0]`; it is not globally forced to `0.9`.
Roban selects the new generic `prev_action_obs=applied` source, so its eight-frame
history contains the EMA-filtered actions actually sent to the simulator. G1 and
L7 retain their existing `mean` history source.

## Tracking bodies and feet

The first motion contract requires root/chest, pelvis, head, hands, upper
arm/forearm and thigh/shin/foot bodies that are verified in the MJCF. Keypoint
tracking excludes root/pelvis because root tracking is already represented by
dedicated observations and rewards.

Roban has no independent toe body. `feet_air_time_ref_dense` is disabled only
in the Roban profile, and disabled rewards skip optional body resolution. G1
and L7 retain the enabled toe reward. Contact sensor primary bodies come from
the profile's `feet_pattern`; Roban uses
`^(leg_l6_link|leg_r6_link)$` without a robot-name branch.

## Student observation and latent

The Student reference steps are exactly:

`[0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -4, -8]`

With `J=21` joints, `S=13` reference frames, history taps
`[0,1,2,3,4,8,12,16,20]` (`H=9`), and `A=8` previous filtered actions, the
existing observation concatenation is already deployment-compatible:

| Observation component | Dimension |
|---|---:|
| Boot indicator | 1 |
| Relative root position, excluding frame 0 | `(S-1)*3 = 36` |
| Root orientation Rot6D | `S*6 = 78` |
| Reference joint position | `S*J = 273` |
| Reference minus current joint position | `S*J = 273` |
| Reference root z | `S = 13` |
| Reference projected gravity | `S*3 = 39` |
| Root angular velocity history | `H*3 = 27` |
| Projected gravity history | `H*3 = 27` |
| Joint position history | `H*J = 189` |
| Joint velocity history | `H*J = 189` |
| Previous filtered actions | `A*J = 168` |
| **Total** | **1313** |

The Roban profile selects `latent_dim=40` and enables the parameter-free
Student latent quantizer with scale `4.499549865722656`:

`y = tanh(priv_pred) * scale`

`student_priv_feature = y + (round(y) - y).detach()`

This yields levels `{-4, …, 4}` and preserves straight-through gradients. The
Student actor consumes `1313 + 40 = 1353` inputs and emits 21 actions. The
teacher actor continues to consume continuous `priv_feature`; estimator MSE is
still between continuous `priv_pred` and continuous teacher `priv_feature`.
The quantizer defaults to disabled and the global latent default remains 256,
preserving G1/L7 and old checkpoint behavior.

## Evidence and provisional values

Verified from the successful Leju controller/config and Student ONNX:

- canonical joint order and hardware-only waist direction;
- zero default pose, action scale, clip, residual mode and action EMA;
- PD/effort values;
- 13 future steps, nine history taps, eight previous actions and 1313 input;
- 40-dimensional estimator, exact quantizer scale/graph, 1353 actor input and
  21-dimensional output.

Verified from the current MJCF compile:

- root/body names, active joints, source actuator order;
- root height, ranges, armature/damping/friction loss;
- collision shapes and `40.69991336 kg` mass.

First-version provisional values borrowed conservatively from G1/L7:

- standing thresholds, termination thresholds and observation noise;
- motion and motor domain randomization, push/wrench, base foot friction and
  friction-randomization ranges;
- training communication-delay, boot-delay and torque-limit curriculum values;
- `soft_joint_pos_limit_factor=0.9`.

## Deferred work and blockers

The data stage is intentionally not implemented. A future `roban/seed/all`
path remains declarative and does not exist yet. Before tracking env startup,
the data stage must validate dataset `joint_names`/`body_names`, Roban foot
names, the hard-coded foot names in `active_adaptation/utils/motion.py`, NPZ and
memdataset production, and 50 Hz motion semantics. No placeholder dataset,
ProtoMotions conversion or loader change is part of this version.

The export stage must still verify ONNX input name `policy`, shape `[1,1313]`,
output name `action`, shape `[1,21]`, embedded observation normalization, the
Tanh/Round quantizer graph, and external-data packaging. Flow Planner,
`best.onnx`, the 27-dimensional flow state, hardware validation and Sim2Real
are out of scope.
