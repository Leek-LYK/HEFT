#!/usr/bin/env bash
# Full six-category Roban Teacher, with global motion-uniform shard weights.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MEMPATH=${MEMPATH:-$REPO_ROOT/dataset}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-roban-full-train}
export PYTHONUNBUFFERED=1
MODE=${MODE:-teacher}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-$REPO_ROOT/.venv/bin/torchrun}
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || ${NPROC:-1} != 1 ]]; then
  echo "Full B200 launcher requires one visible GPU and NPROC=1" >&2
  exit 1
fi
if [[ "$MODE" != teacher && "$MODE" != pipeline ]]; then
  echo "MODE must be teacher or pipeline" >&2
  exit 1
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 1
fi

DATASET_ROOT=${DATASET_ROOT:-$MEMPATH/roban/full/train_v1}
PER_RANK_ENVS=${PER_RANK_ENVS:-16384}
ROLLOUTS=${ROLLOUTS:-50000}
STOP_AFTER_ROLLOUTS=${STOP_AFTER_ROLLOUTS:-null}
CHECKPOINT=${CHECKPOINT:-null}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
OUTPUT=${OUTPUT:-$MEMPATH/_audit/roban_ominisoma/teacher_full_r${ROLLOUTS}_$(date +%Y%m%d_%H%M%S)}
for name in PER_RANK_ENVS ROLLOUTS SAVE_INTERVAL; do
  if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid positive integer: $name=${!name}" >&2
    exit 1
  fi
done
TOTAL_FRAMES=$((PER_RANK_ENVS * 32 * ROLLOUTS))

DATASET_GROUPS=$(.venv/bin/python - "$DATASET_ROOT" <<'PY'
import json
import sys
import yaml
from pathlib import Path
root = Path(sys.argv[1])
report = json.loads((root / 'full_report.json').read_text())
assert report['status'] == 'PASS' and report['source_hashes_unchanged']
groups = json.loads((root / 'dataset_groups.json').read_text())
assert len(groups) == len(report['shards'])
assert sum(g['weight'] for g in groups) == report['motions']
print(yaml.safe_dump(groups, default_flow_style=True, width=1000000, sort_keys=False).strip())
PY
)
if [[ "$MODE" == pipeline ]]; then
  if [[ "$CHECKPOINT" != null ]]; then
    echo "Pipeline uses START_STAGE and CHECKPOINT_PATH, not CHECKPOINT" >&2
    exit 1
  fi
  export NPROC=1 PER_RANK_ENVS STOP_AFTER_ROLLOUTS
  export DATASET_GROUPS_FILE="$DATASET_ROOT/dataset_groups.json"
  export TRAIN_ROLLOUTS=${TRAIN_ROLLOUTS:-$ROLLOUTS}
  export TRAIN_SAVE_INTERVAL=${TRAIN_SAVE_INTERVAL:-$SAVE_INTERVAL}
  exec bash "$REPO_ROOT/train.sh"
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists; choose a new OUTPUT: $OUTPUT" >&2
  exit 1
fi
cmd=("$TORCHRUN" --standalone --nproc_per_node=1 scripts/train.py \
  +exp=roban_train task.num_envs="$PER_RANK_ENVS" \
  "task.profile.command.dataset.groups=$DATASET_GROUPS" \
  +task.profile.command.dataset.resample_on_reset=true \
  task.profile.command.dataset.keep_on_cpu=true \
  total_frames="$TOTAL_FRAMES" stop_after_rollouts="$STOP_AFTER_ROLLOUTS" \
  checkpoint_path="$CHECKPOINT" save_interval="$SAVE_INTERVAL" \
  wandb.mode=disabled tensorboard.enabled=true tensorboard.log_dir="$OUTPUT/tensorboard" \
  train_record.enabled=false headless=true app.headless=true app.enable_cameras=false \
  eval_render=false hydra.run.dir="$OUTPUT" hydra.job.chdir=false hydra.output_subdir=null)
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi
mkdir -p "$OUTPUT"
env -u LD_LIBRARY_PATH "${cmd[@]}" 2>&1 | tee "$OUTPUT/console.log"
