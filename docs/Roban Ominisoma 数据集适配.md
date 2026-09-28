# Roban Ominisoma 数据集适配

本文档定义了旧版 Ominisoma/IsaacLab Roban 导出与 HEFT 当前 Roban 资产之间的转换边界。它不改变 G1/L7 运动契约。

## Existing HEFT pipeline

`scripts/data_process/generate_dataset.py` 接受的标准输入是一个 50 Hz 的 NPZ，包含以下字段：

| Field | Contract |
| --- | --- |
| `root_pos` | `[T, 3]`, world position |
| `root_rot` | `[T, 4]`, quaternion in **xyzw** order |
| `dof_pos` | `[T, J]` |
| `local_body_pos` | `[T, B, 3]`, positions expressed in the rotated root frame |
| `joint_names` | 前 `J` 个 DOF 列的有序名称 |
| `body_names` |  `B` 个 body 列的有序名称 |
| `mocap_framerate`, `frequency`, or `fps` | must resolve to 50 |

标准加载器将 root_rot 从 `xyzw` 恰好转换一次为 `wxyz`，并形成内部 MuJoCo qpos 如下：

```text
[root_xyz, root_quaternion_wxyz, joint_position]
```

它为过滤重建源世界位置如下：

```text
body_pos_w = R_world_root @ local_body_pos + root_pos_w
```
写入器仅在 `_tensordict` 中存储 `root_pos_w`、`root_quat_w（wxyz）`和 `joint_pos`。`meta_motion.json` 包含 `joint_names`、旧版 `body_names`、运动 `starts/ends` 以及可选的逐运动信息。`id_label.json` 包含源/片段标签。不存储根/身体速度和身体位姿。

运行时，`ProgressiveMultiMotionDataset` 从当前资产选择身体名称，`MotionFKHelper` 从三个最小张量重建所有身体位姿和有限差分速度。存储的源身体位姿从不作为运行时目标使用。接受并组合多个配置的数据集路径，但当前加载器将每个分片转换为 float32，然后使用 `torch.cat`。`MotionTrackingCommand` 在低于 1024 个环境时选择 CPU 存储，在 1024 个环境或以上时选择仿真设备。

现有写入器同样为每个被接受的 motion 保留一个张量块，并执行最终 torch.cat；因此无界转换不适合完整 Roban 语料库。

## Ominisoma 源模式

每个已审计源文件使用：

| Field | Shape | Dtype |
| --- | --- | --- |
| `body_ang_vel_w` | `[T, 28, 3]` | `float32` |
| `body_lin_vel_w` | `[T, 28, 3]` | `float32` |
| `body_pos_w` | `[T, 28, 3]` | `float32` |
| `body_quat_w` | `[T, 28, 4]` | `float32`, **wxyz** |
| `joint_pos` | `[T, 21]` | `float32` |
| `joint_vel` | `[T, 21]` | `float32` |
| `fps` | `[1]` | `int64`, value 50 |

根候选是 `base_link`，身体索引 0。源不携带关节/身体名称，因此以下固定契约是适配器的一部分。

### 关节顺序

源顺序：

```text
waist_yaw,
zarm_l1, zarm_r1, leg_l1, leg_r1,
zarm_l2, zarm_r2, leg_l2, leg_r2,
zarm_l3, zarm_r3, leg_l3, leg_r3,
zarm_l4, zarm_r4, leg_l4, leg_r4,
leg_l5, leg_r5, leg_l6, leg_r6
```

目标顺序直接导入自 `ROBAN_JOINT_ORDER`:

```text
waist_yaw_joint,
leg_l1_joint, leg_l2_joint, leg_l3_joint, leg_l4_joint, leg_l5_joint, leg_l6_joint,
leg_r1_joint, leg_r2_joint, leg_r3_joint, leg_r4_joint, leg_r5_joint, leg_r6_joint,
zarm_l1_joint, zarm_l2_joint, zarm_l3_joint, zarm_l4_joint,
zarm_r1_joint, zarm_r2_joint, zarm_r3_joint, zarm_r4_joint
```

由名称推导的源到目标索引为：

```text
(0, 3, 7, 11, 15, 17, 19, 4, 8, 12, 16, 18, 20,
 1, 5, 9, 13, 2, 6, 10, 14)
```

位置和速度数组都使用该置换。缺失、重复或未知的契约名称都是错误。腰部偏航不取反。

### Body 顺序

```text
base_link, waist_yaw_link, zarm_l1_link, zarm_r1_link, zhead_1_link,
leg_l1_link, leg_r1_link, zarm_l2_link, zarm_r2_link, zhead_2_link,
head_radar, leg_l2_link, leg_r2_link, zarm_l3_link, zarm_r3_link,
camera_base, leg_l3_link, leg_r3_link, zarm_l4_link, zarm_r4_link,
leg_l4_link, leg_r4_link, zarm_l5_link, zarm_r5_link,
leg_l5_link, leg_r5_link, leg_l6_link, leg_r6_link
```

`zarm_l5_link`, `zarm_r5_link`, and `camera_base` 属于旧版资产。它们不会被重命名为 `l_handball`, `r_handball`, or `head_camera_base`。源世界身体状态仅用于质量/FK 审计和预处理过滤器。 Roban 适配器写入一个空的旧版 `body_names` 元数据列表；运行时身体状态和名称完全来自当前 MJCF FK。

配置的源脚是 `leg_l6_link` and `leg_r6_link`. 通用预处理器现在接受 `foot_names`; 其默认值仍为
`left_ankle_roll_link` and `right_ankle_roll_link` for G1/L7.

## 四元数与局部位置边界

源四元数始终解释为 wxyz。适配器检查有限值和非零范数，拒绝最大范数偏差超过 `1e-2`, 对可接受值进行归一化，统计相邻负点积，并沿时间应用物理等价的 `q/-q` 符号校正。它从不探测第二种顺序。

如果需要标准格式的 xyzw 值，转换精确为：

```text
root_rot_xyzw = root_quat_wxyz[..., [1, 2, 3, 0]]
```

逆转换经过逐元素测试。适配器可以直接消费世界位置，因此不会制造第二份标准 NPZ。对于需要局部位置的调用方，正确方程为：

```text
local_body_pos = R_world_root.T @ (body_pos_w - root_pos_w)
body_pos_w = root_pos_w + R_world_root @ local_body_pos
```

双向以及非单位旋转根的往返均由测试覆盖。

## FK 审计结果

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

## Smoke results

The single-file smoke used
`main/210531/walk_forward_confident_003__A001.npz`: an ordinary 1,143-frame
forward walk with root height 0.7431–0.8011 m, no joint-range violations, no
NaN/Inf, and no existing-filter rejection. It produced one 1,143-frame motion.
The source SHA-256 matched before and after conversion. Loading through the
normal `MEMPATH` resolver succeeded, and a 16-frame window expanded through the
current MJCF with all fields finite.

The deterministic subset used the first 50 sorted `main` files. All 50 passed,
with 33,574 input/output frames, zero failures, and zero filter skips. The
output was about 1.9 MiB. Random sampling of 16 motions by 12 frames and
batched FK over all 37 non-world current MJCF bodies were finite. Dataset load
and release did not change the process file-descriptor count. Re-running the
same output without `--overwrite` was rejected.

The measured process peak RSS was about 1.5 GiB for both conversions; most of
that baseline comes from importing the project/Torch/MJLab stack rather than
the small stored tensors.

## Full conversion and runtime risks

The `main` category contains 108,193 files and about 62 GiB of compressed
input. Do not pass it unbounded to the current writer. The immediately usable
strategy is explicit 1–2 million-frame memdataset shards, each created with a
bounded `--start-index`, `--max-files`, and `--max-total-frames`. The current
multi-path loader can read such shards only when every shard path is listed
explicitly as a dataset group/path; it does not recursively discover a shard
directory.

Sharding bounds conversion-time `torch.cat` memory and provides atomic resume
points, but it does not solve training-time memory: `load_from_mem_paths`
converts each shard to float32 and concatenates all shards. At 1024 or more
environments, that combined dataset is selected for GPU storage. Before full
training, implement a lazy memmap/shard index (or a two-pass, preallocated
single-dataset writer plus lazy loader) and make dataset placement explicitly
configurable, defaulting large datasets to CPU. Dataset group weights must also
be chosen deliberately if shards are exposed as separate groups.

## Not covered

This stage does not convert the complete corpus, start PPO, enable paired PMG,
change the 40-dimensional quantized Student latent contract, export ONNX, or
perform Sim2Real. The tracking env GPU smoke is conditional on an actually idle
GPU; CPU memdataset and current-MJCF FK checks do not substitute for a simulator
reset/step test.
