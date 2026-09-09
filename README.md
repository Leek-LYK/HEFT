# Motion Tracking

[![HEFT Website](https://img.shields.io/badge/Website-heft.axell.top-0A66C2)](https://heft.axell.top/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![mjlab 1.3.0](https://img.shields.io/badge/mjlab-1.3.0-orange.svg)

This repository contains training, evaluation, export, and deployment-facing assets for
humanoid whole-body motion tracking.

> **Branch guide**
>
> - The **main branch** is the official implementation of
>   [HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion
>   Guidance and Windowed Payload Curriculum](https://heft.axell.top/).
> - The **compliance branch** contains a G1 tracking + compliance framework inspired by
>   Gentle humanoid. [Demo](https://motion-tracking.axell.top/).
> - The **sim2real branch** provides the deployment runtime and available checkpoints.

The simulation and training utilities are based on **mjlab**.

## Repository Structure

To train HEFT G1/L7 policies, clone the default `main` branch:

```bash
git clone https://github.com/Axellwppr/motion_tracking.git
cd motion_tracking
```

To deploy released policies, clone the `sim2real` branch:

```bash
git clone -b sim2real --single-branch https://github.com/Axellwppr/motion_tracking.git motion_tracking_sim2real
cd motion_tracking_sim2real
```

To use the G1 compliance framework, clone the `compliance` branch:

```bash
git clone -b compliance --single-branch https://github.com/Axellwppr/motion_tracking.git motion_tracking_compliance
cd motion_tracking_compliance
```

## Release Status

HEFT consists of an efficient humanoid motion tracking framework, PMG
(Privileged Motion Guidance), and WPC (Windowed Payload Curriculum).

- [x] Efficient motion tracking training framework
- [x] PMG (Privileged Motion Guidance) support
- [x] WPC (Windowed Payload Curriculum) support
- [x] Released checkpoints on the `sim2real` branch
- [x] Deployment runtime on the `sim2real` branch
- [ ] Public training datasets and WPC window-payload labels
- [ ] VR data recording, reference reconstruction, and paired-dataset generation workflow

## Installation

This project uses `uv` for Python dependency and environment management. If you do not
have `uv` installed, follow the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

## Dataset Layout

Training loads motion memdatasets from `dataset/` by default. You can override the
dataset root with the `MEMPATH` environment variable:

```bash
export MEMPATH=/path/to/motion_tracking_dataset
```

The profile files reference paths relative to `MEMPATH`, for example:

```text
dataset/
  g1/
    lafan/
    100style/
    seed/all/
    vr_paired/
      clean/
      raw/
  l7/
    lafan/
    100style/
    seed/all/
    vr_paired/
      clean/
      raw/
```

The paired `vr_paired/{clean,raw}` folders are used by PMG. The clean side is used as
the teacher/reference target, while the raw side represents deployable student input.

We provide some preprocessed G1 memdatasets for smoke-testing the
training pipeline:
[Google Drive dataset](https://drive.google.com/drive/folders/1-FBUxllaYwqGIUSCaWg_4inD-u5Tdvi9?usp=sharing).
After downloading, place the extracted folders under `dataset/g1/` or set `MEMPATH` to
the extracted dataset root. These samples are not the exact HEFT training set, but they
are useful for validating the framework and obtaining similar training behavior. The
full dataset will be released later.

## Build A Motion Dataset

The dataset builder consumes retargeted robot `npz` files and writes a memdataset:

Each `npz` should follow the GMR-style robot motion format at 50 Hz. The required
fields are `root_pos`, `root_rot` in `xyzw` quaternion order, `dof_pos`,
`local_body_pos`, `joint_names`, and `body_names`. You can use the
[modified GMR exporter](https://github.com/Axellwppr/GMR) to generate compatible
retargeted motions.

```bash
uv run python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/retargeted_npz \
  --mem-path dataset/g1/my_motion_set
```

For paired PMG data, provide both clean teacher motions and raw student motions:

```bash
uv run python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/clean_npz \
  --mem-path dataset/g1/vr_paired/clean \
  --student-root /path/to/raw_npz \
  --student-mem-path dataset/g1/vr_paired/raw
```

If you write to `dataset/g1/my_motion_set`, use `g1/my_motion_set` in the YAML profile.
If you write under another root, set `MEMPATH` before training.

## Configure Training

Robot-specific training settings live in:

- `cfg/task/profile/G1_tracking.yaml`
- `cfg/task/profile/L7_tracking.yaml`

The most common changes are:

- dataset groups under `command.dataset.groups`
- dataset weights
- whether the `vr_paired` group is enabled
- `student_motion_randomization.enable`
- action/noise/randomization ranges for a specific robot

For PMG training with paired clean/raw data:

1. Enable the `vr_paired` dataset group.
2. Set `student_motion_randomization.enable: false`.

If paired PMG data is unavailable:

1. Leave the paired group commented out.
2. Keep `student_motion_randomization.enable: true`.

L7 WPC support is configured through `cfg/load/l7_wpc.yaml`, but the public WPC payload
labels and datasets are planned for a later release.

## Training

The local `train.sh` is configured for Roban and the 50K motion dataset. Run all
three stages with one command:

```bash
bash train.sh
```

Defaults: one B200 GPU (`0`), 16,384 environments, Teacher 100,000 rollouts,
Adapt 10,000 rollouts, Finetune 10,000 rollouts. These are budgets, not convergence
guarantees. The stages use `roban_train`, `roban_adapt`, and `roban_finetune`.
Each successful stage passes its local `checkpoint_final.pt` to the next stage.
A failed command or missing checkpoint stops the pipeline.

W&B is disabled. Each stage has its own `cfg.yaml`, checkpoints, console log,
command record and `tensorboard/` under
`dataset/_audit/roban_ominisoma/roban_pipeline_<timestamp>/{train,adapt,finetune}`.
Teacher checkpoints are saved every 1,000 rollouts; Adapt/Finetune every 250.
The final Student checkpoint is for evaluation and export; this script does not
start deployment or export ONNX automatically.

Preview without starting training or creating output files:

```bash
DRY_RUN=1 bash train.sh
```

Override GPUs, budgets, dataset or output directory with environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 PER_RANK_ENVS=16384 \
TRAIN_ROLLOUTS=100000 ADAPT_ROLLOUTS=10000 FINETUNE_ROLLOUTS=10000 \
DATASET=roban/main/train_50000_v1 OUTPUT_ROOT=/absolute/path/to/new_run \
bash train.sh
```

Start at Adapt using an existing Teacher, or at Finetune using an existing Adapt
checkpoint (always choose a new output root):

```bash
START_STAGE=adapt CHECKPOINT_PATH=/absolute/path/to/teacher/checkpoint_final.pt bash train.sh
START_STAGE=finetune CHECKPOINT_PATH=/absolute/path/to/adapt/checkpoint_final.pt bash train.sh
```

This loads network and normalization state; current checkpoints do not include
optimizer state for exact training resumption. `NPROC` defaults to the number of
listed GPUs. `MEMPATH` defaults to this repository's `dataset` directory.

## Evaluation And Export

Each stage saves a checkpoint every 150 iterations. During or after training, you can
use `eval.py` to evaluate checkpoint performance or export a deployment policy.

Play a trained policy:

```bash
uv run python scripts/eval.py --run_path ${wandb_run_path} -p
```

Export a deployment policy:

```bash
uv run python scripts/eval.py --run_path ${wandb_run_path} -p --export
```

Evaluate with an explicit profile:

```bash
uv run python scripts/eval.py --run_path ${wandb_run_path} --task tracking --profile L7_tracking -p
```

Exported policies are written under:

```text
scripts/exports/<task-name>-<timestamp>/
```

The exported folder contains the policy artifacts needed by the deployment runtime,
including `policy.onnx`, `policy.pt`, and `policy.json`.

## Deployment

Deployment assets are provided on the `sim2real` branch:

```bash
git clone -b sim2real --single-branch https://github.com/Axellwppr/motion_tracking.git motion_tracking_sim2real
```

That branch contains the deployment runtime, sim2sim/sim2real configuration, and
released policy assets.
