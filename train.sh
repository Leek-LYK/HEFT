#!/usr/bin/env bash
# Roban: Teacher -> Student Adapt -> Student Finetune, with local checkpoints.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MEMPATH=${MEMPATH:-$REPO_ROOT/dataset}
export WANDB_MODE=disabled
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-roban-pipeline}
export PYTHONUNBUFFERED=1

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${NPROC:-${#GPU_IDS[@]}}
PER_RANK_ENVS=${PER_RANK_ENVS:-16384}
TRAIN_ROLLOUTS=${TRAIN_ROLLOUTS:-100000}
ADAPT_ROLLOUTS=${ADAPT_ROLLOUTS:-10000}
FINETUNE_ROLLOUTS=${FINETUNE_ROLLOUTS:-10000}
TRAIN_SAVE_INTERVAL=${TRAIN_SAVE_INTERVAL:-1000}
ADAPT_SAVE_INTERVAL=${ADAPT_SAVE_INTERVAL:-250}
FINETUNE_SAVE_INTERVAL=${FINETUNE_SAVE_INTERVAL:-250}
STOP_AFTER_ROLLOUTS=${STOP_AFTER_ROLLOUTS:-null}
DATASET=${DATASET:-roban/main/train_50000_v1}
DATASET_GROUPS_FILE=${DATASET_GROUPS_FILE:-}
if [[ -n "$DATASET_GROUPS_FILE" ]]; then
  DATASET_GROUPS=$("$REPO_ROOT/.venv/bin/python" - "$DATASET_GROUPS_FILE" <<'PY'
import json, math, sys, yaml
from pathlib import Path
p = Path(sys.argv[1])
groups = json.loads(p.read_text())
assert isinstance(groups, list) and groups
for group in groups:
    assert group['teacher_mem_path'] and math.isfinite(group['weight']) and group['weight'] > 0
report = p.parent / 'full_report.json'
if report.exists():
    assert json.loads(report.read_text())['status'] == 'PASS'
else:
    raise ValueError('Full-corpus groups require a completed full_report.json')
print(yaml.safe_dump(groups, default_flow_style=True, width=1000000, sort_keys=False).strip())
PY
)
  RESAMPLE_ON_RESET=${RESAMPLE_ON_RESET:-true}
else
  DATASET_GROUPS="[{name:formal_50k,teacher_mem_path:$DATASET,weight:1.0}]"
  RESAMPLE_ON_RESET=${RESAMPLE_ON_RESET:-false}
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/dataset/_audit/roban_ominisoma/roban_pipeline_$(date +%Y%m%d_%H%M%S)}
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
START_STAGE=${START_STAGE:-train}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-$REPO_ROOT/.venv/bin/torchrun}

for name in NPROC PER_RANK_ENVS TRAIN_ROLLOUTS ADAPT_ROLLOUTS FINETUNE_ROLLOUTS TRAIN_SAVE_INTERVAL ADAPT_SAVE_INTERVAL FINETUNE_SAVE_INTERVAL; do
  if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid positive integer: $name=${!name}" >&2
    exit 1
  fi
done
if (( NPROC > ${#GPU_IDS[@]} )); then
  echo "NPROC exceeds the number of visible GPUs" >&2
  exit 1
fi
case "$START_STAGE" in
  train) FIRST=0 ;;
  adapt) FIRST=1 ;;
  finetune) FIRST=2 ;;
  *) echo "START_STAGE must be train, adapt or finetune" >&2; exit 1 ;;
esac
if (( FIRST > 0 )); then
  if [[ ! -s "$CHECKPOINT_PATH" ]]; then
    echo "Starting at $START_STAGE requires CHECKPOINT_PATH to an existing checkpoint" >&2
    exit 1
  fi
  [[ "$CHECKPOINT_PATH" = /* ]] || CHECKPOINT_PATH="$REPO_ROOT/$CHECKPOINT_PATH"
elif [[ -n "$CHECKPOINT_PATH" ]]; then
  echo "Use START_STAGE=adapt or finetune to load a previous stage checkpoint" >&2
  exit 1
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1" >&2; exit 1
fi
# Never overwrite an earlier pipeline. Use a new root when starting from a checkpoint.
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2; exit 1
fi
if [[ "$DRY_RUN" == 0 ]]; then
  [[ -x "$TORCHRUN" ]] || { echo "Missing torchrun: $TORCHRUN" >&2; exit 1; }
  mkdir -p "$OUTPUT_ROOT"
  cp "$REPO_ROOT/train.sh" "$OUTPUT_ROOT/pipeline.sh"
fi

STAGES=(train adapt finetune)
ROLLOUTS=("$TRAIN_ROLLOUTS" "$ADAPT_ROLLOUTS" "$FINETUNE_ROLLOUTS")
STEPS=(32 16 32)
SAVE_INTERVALS=("$TRAIN_SAVE_INTERVAL" "$ADAPT_SAVE_INTERVAL" "$FINETUNE_SAVE_INTERVAL")
checkpoint=${CHECKPOINT_PATH:-null}
for (( index=FIRST; index<3; index++ )); do
  stage=${STAGES[index]}
  stage_dir="$OUTPUT_ROOT/$stage"
  total_frames=$((NPROC * PER_RANK_ENVS * STEPS[index] * ROLLOUTS[index]))
  cmd=("$TORCHRUN" --standalone --nproc_per_node="$NPROC" "$REPO_ROOT/scripts/train.py"
    task=tracking "+exp=roban_$stage" "task.num_envs=$PER_RANK_ENVS"
    "task.profile.command.dataset.groups=$DATASET_GROUPS"
    "+task.profile.command.dataset.resample_on_reset=$RESAMPLE_ON_RESET"
    "algo.train_every=${STEPS[index]}" "total_frames=$total_frames" start_iter=0
    "stop_after_rollouts=$STOP_AFTER_ROLLOUTS"
    "checkpoint_path=$checkpoint" "save_interval=${SAVE_INTERVALS[index]}"
    wandb.mode=disabled tensorboard.enabled=true "tensorboard.log_dir=$stage_dir/tensorboard"
    train_record.enabled=false headless=true app.headless=true app.enable_cameras=false eval_render=false
    "hydra.run.dir=$stage_dir" hydra.job.chdir=false hydra.output_subdir=null)
  echo "Stage $stage: ${ROLLOUTS[index]} rollouts, $total_frames global frames"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$stage_dir"
    printf '%q ' "${cmd[@]}" > "$stage_dir/command.txt"
    printf '\n' >> "$stage_dir/command.txt"
    # Use the CUDA libraries bundled with PyTorch, not inherited Conda/system libraries.
    env -u LD_LIBRARY_PATH "${cmd[@]}" 2>&1 | tee "$stage_dir/console.log"
    if [[ ! -s "$stage_dir/checkpoint_final.pt" ]]; then
      echo "Stage $stage did not produce checkpoint_final.pt; stopping pipeline" >&2
      exit 1
    fi
  fi
  checkpoint="$stage_dir/checkpoint_final.pt"
done
if [[ "$DRY_RUN" == 1 ]]; then
  echo "Dry run only; no training started."
else
  echo "Completed. Student checkpoint for evaluation/export: $checkpoint"
fi
