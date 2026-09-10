#!/usr/bin/env bash
# Full six-category Roban Teacher, with global motion-uniform shard weights.
set -euo pipefail
cd /data/liyinkun/project_HEFT
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=3,4,7
export MEMPATH=/data/liyinkun/project_HEFT/dataset
export MPLCONFIGDIR=/tmp/mpl-roban-full-train

DATASET_ROOT=${DATASET_ROOT:-$MEMPATH/roban/full/train_v1}
PER_RANK_ENVS=${PER_RANK_ENVS:-4096}
ROLLOUTS=${ROLLOUTS:-50000}
STOP_AFTER_ROLLOUTS=${STOP_AFTER_ROLLOUTS:-null}
CHECKPOINT=${CHECKPOINT:-null}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
OUTPUT=${OUTPUT:-/data/liyinkun/project_HEFT/dataset/_audit/roban_ominisoma/teacher_full_r50000_v1}
TOTAL_FRAMES=$((PER_RANK_ENVS * 32 * ROLLOUTS * 3))

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
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists; choose a new OUTPUT: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
.venv/bin/torchrun --standalone --nproc_per_node=3 scripts/train.py \
  +exp=roban_train task.num_envs="$PER_RANK_ENVS" \
  "task.profile.command.dataset.groups=$DATASET_GROUPS" \
  +task.profile.command.dataset.resample_on_reset=true \
  task.profile.command.dataset.keep_on_cpu=true \
  total_frames="$TOTAL_FRAMES" stop_after_rollouts="$STOP_AFTER_ROLLOUTS" \
  checkpoint_path="$CHECKPOINT" save_interval="$SAVE_INTERVAL" \
  wandb.mode=disabled tensorboard.enabled=true tensorboard.log_dir="$OUTPUT/tensorboard" \
  train_record.enabled=false headless=true app.headless=true app.enable_cameras=false \
  eval_render=false hydra.run.dir="$OUTPUT" hydra.job.chdir=false hydra.output_subdir=null \
  2>&1 | tee "$OUTPUT/console.log"
