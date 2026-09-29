#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=4
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/refinetrack_v01/launch.py" --formal --condition C1 --stage adapt --gpu 4 --output "${RUN_DIR:?Set a new RUN_DIR}" --parent "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to previous formal selected checkpoint}"
