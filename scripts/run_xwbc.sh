#!/usr/bin/env bash
# Single B200 entry point. Formal runs use direct PPO, without payloads or stages.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset LD_LIBRARY_PATH
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
MODE="${1:-train}"
CONFIG="${CONFIG:-$PROJECT_ROOT/cfg/xwbc/heft_context.yaml}"
DATASET="${DATASET:-$PROJECT_ROOT/dataset/xwbc/paired_10k_v1}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/dataset/xwbc/manifests/paired_10k.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/logs/v0.1/xwbc_$(date -u +%Y%m%dT%H%M%SZ)}"
NUM_ENVS="${NUM_ENVS:-}"
UPDATES="${UPDATES:-}"
VARIANT="${VARIANT:-shared}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

run() {
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}

prepare() {
    if [[ ! -f "$MANIFEST" ]]; then
        run "$PYTHON" -B scripts/data_process/select_xwbc_motions.py --output "$MANIFEST"
    fi
    if [[ ! -f "$DATASET/READY" ]]; then
        run "$PYTHON" -B scripts/data_process/build_xwbc_dataset.py --manifest "$MANIFEST" --output "$DATASET" --workers "${WORKERS:-4}"
    else
        run "$PYTHON" -B scripts/data_process/build_xwbc_dataset.py --manifest "$MANIFEST" --output "$DATASET" --verify-only
    fi
}

train_one() {
    local variant="$1" seed="$2" destination="$3"
    local args=(train --config "$CONFIG" --dataset "$DATASET" --output "$destination" --variant "$variant" --seed "$seed")
    if [[ -n "$NUM_ENVS" ]]; then args+=(--num-envs "$NUM_ENVS"); fi
    if [[ -n "$UPDATES" ]]; then args+=(--updates "$UPDATES"); fi
    if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
    if [[ -n "${MAX_MINUTES:-}" ]]; then args+=(--max-minutes "$MAX_MINUTES"); fi
    if [[ "$DRY_RUN" == 1 ]]; then
        run "$PYTHON" -u -B scripts/xwbc.py "${args[@]}"
    else
        mkdir -p "$(dirname -- "$destination")"
        run "$PYTHON" -u -B scripts/xwbc.py "${args[@]}" 2>&1 | tee "$destination.console.log"
    fi
}

case "$MODE" in
    prepare) prepare ;;
    smoke)
        if [[ ! -f "$MANIFEST" ]]; then
            run "$PYTHON" -B scripts/data_process/select_xwbc_motions.py --output "$MANIFEST"
        fi
        DATASET="${SMOKE_DATASET:-$PROJECT_ROOT/dataset/xwbc/smoke_v1}"
        if [[ ! -f "$DATASET/READY" ]]; then
            run "$PYTHON" -B scripts/data_process/build_xwbc_dataset.py --manifest "$MANIFEST" --output "$DATASET" --limit-per-split 12 --workers 2
        fi
        NUM_ENVS="${SMOKE_ENVS:-32}"
        UPDATES="${SMOKE_UPDATES:-2}"
        MAX_MINUTES="${MAX_MINUTES:-10}"
        train_one "$VARIANT" "$SEED" "$OUTPUT_ROOT/train"
        run "$PYTHON" -u -B scripts/xwbc.py evaluate --checkpoint "$OUTPUT_ROOT/train/checkpoint_final.pt" --dataset "$DATASET" --output "$OUTPUT_ROOT/eval_val" --split val --num-envs 4 --max-motions 4
        ;;
    train)
        prepare
        train_one "$VARIANT" "$SEED" "$OUTPUT_ROOT"
        ;;
    evaluate)
        : "${CHECKPOINT:?Set CHECKPOINT to an XWBC checkpoint}"
        run "$PYTHON" -u -B scripts/xwbc.py evaluate --checkpoint "$CHECKPOINT" --dataset "$DATASET" --output "$OUTPUT_ROOT" --split "${SPLIT:-test}" --num-envs "${EVAL_ENVS:-64}"
        ;;
    suite)
        prepare
        if [[ -n "${RESUME:-}" ]]; then printf 'Use train mode for a single resume.\n' >&2; exit 2; fi
        for variant in ${VARIANTS:-shared independent no_align}; do
            for seed in ${SEEDS:-0 1 2}; do
                destination="$OUTPUT_ROOT/${variant}_seed${seed}"
                train_one "$variant" "$seed" "$destination"
                for split in val test; do
                    run "$PYTHON" -u -B scripts/xwbc.py evaluate --checkpoint "$destination/checkpoint_final.pt" --dataset "$DATASET" --output "$destination/eval_$split" --split "$split" --num-envs "${EVAL_ENVS:-64}"
                done
            done
        done
        run "$PYTHON" -B scripts/summarize_xwbc.py --root "$OUTPUT_ROOT"
        ;;
    *) printf 'Usage: bash scripts/run_xwbc.sh {prepare|smoke|train|evaluate|suite}\n' >&2; exit 2 ;;
esac
