#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=7
: "${C0_CHECKPOINT:?Set C0 formal final checkpoint}"
: "${C1_CHECKPOINT:?Set C1 formal final checkpoint}"
: "${RUN_DIR:?Set new evaluation root}"
[[ ! -e "$RUN_DIR" ]] || { echo 'Output exists' >&2; exit 1; }
mkdir -p "$RUN_DIR"
for condition in C0 C1; do
  key="${condition}_CHECKPOINT"
  for severity in S0 S1 S2 S3; do
    "$ROOT/.venv/bin/python" scripts/refinetrack_v01/evaluate.py --checkpoint "${!key}" --split test --severity "$severity" --output "$RUN_DIR/${condition}_${severity}" > "$RUN_DIR/${condition}_${severity}.log" 2>&1
  done
  for severity in S1 S2 S3; do
    "$ROOT/.venv/bin/python" scripts/refinetrack_v01/evaluate.py --checkpoint "${!key}" --split test --recipe heldout --severity "$severity" --output "$RUN_DIR/${condition}_heldout_${severity}" > "$RUN_DIR/${condition}_heldout_${severity}.log" 2>&1
  done
done
for severity in S0 S1 S2 S3 heldout_S1 heldout_S2 heldout_S3; do
  "$ROOT/.venv/bin/python" scripts/refinetrack_v01/compare.py --left "$RUN_DIR/C0_${severity}" --right "$RUN_DIR/C1_${severity}" --output "$RUN_DIR/comparison_${severity}.json"
done
