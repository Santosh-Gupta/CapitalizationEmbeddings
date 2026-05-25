#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/repos/CapitalizationEmbeddings}"
WORK_ROOT="${CAP_EMB_WORK_ROOT:-/workspace/capitalization_embeddings}"
PYTHON="${PYTHON:-python}"

cd "$REPO_ROOT"
export CAP_EMB_WORK_ROOT="$WORK_ROOT"

V3_ROOT="${V3_ROOT:-$WORK_ROOT/checkpoints/mlm/v3}"

UNC_INITIAL="${UNC_INITIAL:-}"
CAS_INITIAL="${CAS_INITIAL:-}"
CAP_INITIAL="${CAP_INITIAL:-}"

WARMUP_STEPS="${WARMUP_STEPS:-1000}"
GENERAL_STEPS="${GENERAL_STEPS:-20000}"
DOMAIN_STEPS="${DOMAIN_STEPS:-5000}"
MAX_LENGTH="${MAX_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
GENERAL_LR="${GENERAL_LR:-2e-5}"
DOMAIN_LR="${DOMAIN_LR:-1e-5}"
WARMUP_LR="${WARMUP_LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
CAP_CLASS_WEIGHTS="${CAP_CLASS_WEIGHTS:-1,2,8,4}"

run_mlm() {
  local model_kind="$1"
  local initial_checkpoint="$2"
  local corpus="$3"
  local output_root="$4"
  local max_steps="$5"
  local learning_rate="$6"
  shift 6

  if [[ -f "$output_root/final/config.json" ]]; then
    echo "SKIP $output_root: final checkpoint already exists"
    return
  fi

  "$PYTHON" -u scripts/run_mlm_pretraining.py \
    --model-kind "$model_kind" \
    --initial-checkpoint "$initial_checkpoint" \
    --corpus "$corpus" \
    --output-root "$output_root" \
    --max-steps "$max_steps" \
    --max-length "$MAX_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$learning_rate" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup-ratio "$WARMUP_RATIO" \
    "$@"
}

echo "== V3 Stage 0: matched general-corpus warmup =="
echo "Baselines receive same-step MLM continuation; cap model uses a cap-focused warmup."
run_mlm uncased "$UNC_INITIAL" capitalization_v3_general \
  "$V3_ROOT/uncased_warmup_general_steps${WARMUP_STEPS}_lr${WARMUP_LR}" \
  "$WARMUP_STEPS" "$WARMUP_LR"
run_mlm cased "$CAS_INITIAL" capitalization_v3_general \
  "$V3_ROOT/cased_warmup_general_steps${WARMUP_STEPS}_lr${WARMUP_LR}" \
  "$WARMUP_STEPS" "$WARMUP_LR"
run_mlm capitalized "$CAP_INITIAL" capitalization_v3_general \
  "$V3_ROOT/capitalized_warmup_general_steps${WARMUP_STEPS}_lr${WARMUP_LR}" \
  "$WARMUP_STEPS" "$WARMUP_LR" \
  --use-mixed-case-capitalization \
  --capitalization-loss-weight 1.0 \
  --capitalization-class-weights "$CAP_CLASS_WEIGHTS" \
  --capitalization-embedding-dropout 0.0 \
  --freeze-non-capitalization-parameters

UNC_WARM="$V3_ROOT/uncased_warmup_general_steps${WARMUP_STEPS}_lr${WARMUP_LR}/final"
CAS_WARM="$V3_ROOT/cased_warmup_general_steps${WARMUP_STEPS}_lr${WARMUP_LR}/final"
CAP_WARM="$V3_ROOT/capitalized_warmup_general_steps${WARMUP_STEPS}_lr${WARMUP_LR}/final"

echo "== V3 Stage 1: matched general natural-casing continuation =="
run_mlm uncased "$UNC_WARM" capitalization_v3_general \
  "$V3_ROOT/uncased_general_steps${GENERAL_STEPS}_lr${GENERAL_LR}" \
  "$GENERAL_STEPS" "$GENERAL_LR"
run_mlm cased "$CAS_WARM" capitalization_v3_general \
  "$V3_ROOT/cased_general_steps${GENERAL_STEPS}_lr${GENERAL_LR}" \
  "$GENERAL_STEPS" "$GENERAL_LR"
run_mlm capitalized "$CAP_WARM" capitalization_v3_general \
  "$V3_ROOT/capitalized_general_steps${GENERAL_STEPS}_lr${GENERAL_LR}_drop005" \
  "$GENERAL_STEPS" "$GENERAL_LR" \
  --use-mixed-case-capitalization \
  --capitalization-loss-weight 0.5 \
  --capitalization-class-weights "$CAP_CLASS_WEIGHTS" \
  --capitalization-embedding-dropout 0.05

UNC_GEN="$V3_ROOT/uncased_general_steps${GENERAL_STEPS}_lr${GENERAL_LR}/final"
CAS_GEN="$V3_ROOT/cased_general_steps${GENERAL_STEPS}_lr${GENERAL_LR}/final"
CAP_GEN="$V3_ROOT/capitalized_general_steps${GENERAL_STEPS}_lr${GENERAL_LR}_drop005/final"

echo "== V3 Stage 2: matched train-split domain-adaptive continuation =="
run_mlm uncased "$UNC_GEN" capitalization_v3_domain_train \
  "$V3_ROOT/uncased_domain_steps${DOMAIN_STEPS}_lr${DOMAIN_LR}" \
  "$DOMAIN_STEPS" "$DOMAIN_LR"
run_mlm cased "$CAS_GEN" capitalization_v3_domain_train \
  "$V3_ROOT/cased_domain_steps${DOMAIN_STEPS}_lr${DOMAIN_LR}" \
  "$DOMAIN_STEPS" "$DOMAIN_LR"
run_mlm capitalized "$CAP_GEN" capitalization_v3_domain_train \
  "$V3_ROOT/capitalized_domain_steps${DOMAIN_STEPS}_lr${DOMAIN_LR}_drop005" \
  "$DOMAIN_STEPS" "$DOMAIN_LR" \
  --use-mixed-case-capitalization \
  --capitalization-loss-weight 0.25 \
  --capitalization-class-weights "$CAP_CLASS_WEIGHTS" \
  --capitalization-embedding-dropout 0.05

echo "V3 final checkpoints:"
echo "uncased=$V3_ROOT/uncased_domain_steps${DOMAIN_STEPS}_lr${DOMAIN_LR}/final"
echo "cased=$V3_ROOT/cased_domain_steps${DOMAIN_STEPS}_lr${DOMAIN_LR}/final"
echo "capitalized=$V3_ROOT/capitalized_domain_steps${DOMAIN_STEPS}_lr${DOMAIN_LR}_drop005/final"
