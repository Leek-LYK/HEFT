#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset LD_LIBRARY_PATH
out="logs/v0.2/refinement_20261005/teacher_correction_run01"
.venv/bin/python -u -B scripts/refine_xwbc_student.py --output "$out"
for iteration in 100 200 300; do
  .venv/bin/python -u -B scripts/xwbc_staged.py evaluate \
    --checkpoint "$out/checkpoint_${iteration}.pt" \
    --output "$out/eval_val_${iteration}" --num-envs 64 --split val
done
