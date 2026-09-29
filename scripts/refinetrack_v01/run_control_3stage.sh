#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=3
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/refinetrack_v01/pipeline.py" --formal --condition C0 --gpu 3 --output "${RUN_DIR:?Set a new RUN_DIR}"
