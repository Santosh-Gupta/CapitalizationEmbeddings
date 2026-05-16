#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/repos/CapitalizationEmbeddings}"
WORK_ROOT="${CAP_EMB_WORK_ROOT:-/workspace/capitalization_embeddings}"
PYTHON="${PYTHON:-python}"

cd "$REPO_ROOT"
export CAP_EMB_WORK_ROOT="$WORK_ROOT"

REAL_ACRONYM_ROOT="$WORK_ROOT/checkpoints/mlm/real_acronym_mix"
CURRENT_CAP="$WORK_ROOT/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final"

UNC_ROUND1="$REAL_ACRONYM_ROOT/uncased_from_task_mix_steps3000_lr2e5/final"
CAS_ROUND1="$REAL_ACRONYM_ROOT/cased_from_task_mix_steps3000_lr2e5/final"
UNC_ROUND2="$REAL_ACRONYM_ROOT/uncased_round2_steps3000_lr2e5/final"
CAS_ROUND2="$REAL_ACRONYM_ROOT/cased_round2_steps3000_lr2e5/final"

V2_ROOT="$WORK_ROOT/checkpoints/mlm/domain_mix_v2"
UNC_V2="$V2_ROOT/uncased_from_round2_steps3000_lr2e5"
CAS_V2="$V2_ROOT/cased_from_round2_steps3000_lr2e5"
CAP_V2="$V2_ROOT/capitalized_from_mixed_case_current_steps3000_lr2e5"

run_mlm() {
  local model_kind="$1"
  local initial_checkpoint="$2"
  local corpus="$3"
  local output_root="$4"
  shift 4

  if [[ -f "$output_root/final/config.json" ]]; then
    echo "SKIP $output_root: final checkpoint already exists"
    return
  fi

  "$PYTHON" -u scripts/run_mlm_pretraining.py \
    --model-kind "$model_kind" \
    --initial-checkpoint "$initial_checkpoint" \
    --corpus "$corpus" \
    --output-root "$output_root" \
    --max-steps 3000 \
    --batch-size 32 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.06 \
    "$@"
}

echo "== Compute-matching cased/uncased controls with a second real-acronym pass =="
run_mlm uncased "$UNC_ROUND1" capitalization_real_acronym_mix "$REAL_ACRONYM_ROOT/uncased_round2_steps3000_lr2e5"
run_mlm cased "$CAS_ROUND1" capitalization_real_acronym_mix "$REAL_ACRONYM_ROOT/cased_round2_steps3000_lr2e5"

echo "== Running matched capitalization_domain_mix_v2 continuation =="
run_mlm uncased "$UNC_ROUND2" capitalization_domain_mix_v2 "$UNC_V2"
run_mlm cased "$CAS_ROUND2" capitalization_domain_mix_v2 "$CAS_V2"
run_mlm \
  capitalized \
  "$CURRENT_CAP" \
  capitalization_domain_mix_v2 \
  "$CAP_V2" \
  --use-mixed-case-capitalization \
  --capitalization-loss-weight 0.25 \
  --capitalization-class-weights 1,2,8,4 \
  --capitalization-embedding-dropout 0.1

echo "V2 checkpoints:"
echo "uncased=$UNC_V2/final"
echo "cased=$CAS_V2/final"
echo "capitalized=$CAP_V2/final"
