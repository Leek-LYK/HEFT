#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=3
: "${RUN_DIR:?Set a new smoke output root}"
[[ ! -e "$RUN_DIR" ]] || { echo 'Output exists' >&2; exit 1; }
mkdir -p "$RUN_DIR"
for condition in C0 C1; do
  "$ROOT/.venv/bin/python" "$ROOT/scripts/refinetrack_v01/pipeline.py" --condition "$condition" --gpu 3 --output "$RUN_DIR/$condition"
done
