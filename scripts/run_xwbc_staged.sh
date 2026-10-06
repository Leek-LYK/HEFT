#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset LD_LIBRARY_PATH
MODE="${1:-preview}"
if [[ "$MODE" == finetune ]]; then
  CONFIG="${CONFIG:-$PROJECT_ROOT/cfg/xwbc/v02/finetune.yaml}"
else
  CONFIG="${CONFIG:-$PROJECT_ROOT/cfg/xwbc/v02/staged.yaml}"
fi
DATASET="${DATASET:-$PROJECT_ROOT/dataset/xwbc/simple_5k_v02}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/dataset/xwbc/manifests/simple_5k_v02.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/logs/v0.2/$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
run() { printf '%q ' "$@"; printf '\n'; if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi; }
train_args=(--config "$CONFIG" --dataset "$DATASET" --output "$OUTPUT_ROOT")
if [[ -n "${NUM_ENVS:-}" ]]; then train_args+=(--num-envs "$NUM_ENVS"); fi
if [[ -n "${UPDATES:-}" ]]; then train_args+=(--updates "$UPDATES"); fi
if [[ -n "${SEED:-}" ]]; then train_args+=(--seed "$SEED"); fi
if [[ -n "${RESUME:-}" ]]; then train_args+=(--resume "$RESUME"); fi
if [[ -n "${MAX_MINUTES:-}" ]]; then train_args+=(--max-minutes "$MAX_MINUTES"); fi
case "$MODE" in
 prepare)
  if [[ ! -f "$MANIFEST" ]]; then run "$PYTHON" -u -B scripts/data_process/select_xwbc_simple.py --output "$MANIFEST" --workers "${WORKERS:-2}"; fi
  if [[ ! -f "$DATASET/READY" ]]; then
   run "$PYTHON" -u -B scripts/data_process/build_xwbc_dataset.py --manifest "$MANIFEST" --output "$DATASET" --workers "${WORKERS:-2}"
  else
   run "$PYTHON" -B scripts/data_process/build_xwbc_dataset.py --manifest "$MANIFEST" --output "$DATASET" --verify-only
  fi ;;
 teacher)
  : "${ROBOT:?Set ROBOT=g1 or roban}"
  run "$PYTHON" -u -B scripts/xwbc_staged.py teacher --robot "$ROBOT" "${train_args[@]}" ;;
 student)
  : "${G1_TEACHER:?Set approved G1_TEACHER}" "${ROBAN_TEACHER:?Set approved ROBAN_TEACHER}"
  : "${G1_GATE:?Set G1_GATE}" "${ROBAN_GATE:?Set ROBAN_GATE}"
  run "$PYTHON" -u -B scripts/xwbc_staged.py student "${train_args[@]}" --g1-teacher "$G1_TEACHER" --roban-teacher "$ROBAN_TEACHER" --g1-gate "$G1_GATE" --roban-gate "$ROBAN_GATE" ;;
 sources)
  : "${STUDENT:?Set stage B STUDENT checkpoint}" "${STUDENT_GATE:?Set STUDENT_GATE}"
  run "$PYTHON" -u -B scripts/xwbc_staged.py sources "${train_args[@]}" --student "$STUDENT" --student-gate "$STUDENT_GATE" ;;
 finetune)
  : "${POLICY:?Set approved stage C POLICY}" "${POLICY_GATE:?Set POLICY_GATE}"
  run "$PYTHON" -u -B scripts/xwbc_staged.py finetune "${train_args[@]}" --policy "$POLICY" --policy-gate "$POLICY_GATE" ;;
 evaluate)
  : "${CHECKPOINT:?Set CHECKPOINT}"
  args=(--checkpoint "$CHECKPOINT" --output "$OUTPUT_ROOT" --dataset "$DATASET" --split "${SPLIT:-val}" --num-envs "${EVAL_ENVS:-64}")
  if [[ -n "${MAX_MOTIONS:-}" ]]; then args+=(--max-motions "$MAX_MOTIONS"); fi
  run "$PYTHON" -u -B scripts/xwbc_staged.py evaluate "${args[@]}" ;;
 *) printf 'Modes: prepare | teacher | student | sources | finetune | evaluate\nA/B/C require full validation before the next stage. D runs only via explicit finetune.\n' ;;
esac
