#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${CAP_EMB_REPO:-/workspace/repos/CapitalizationEmbeddings}"
WORK_ROOT="${CAP_EMB_WORK_ROOT:-/workspace/capitalization_embeddings}"

cd "$REPO_ROOT"

LOG_ROOT="$WORK_ROOT/logs/ablations_case_channel"
ABLATION_ROOT="$WORK_ROOT/checkpoints/ablations_case_channel_5seed"
mkdir -p "$LOG_ROOT" "$ABLATION_ROOT"

UNC="$WORK_ROOT/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final"
CAS="$WORK_ROOT/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final"
CAP_3CLASS="$WORK_ROOT/checkpoints/mlm/real_acronym_mix/capitalized_from_task_mix_steps3000_lr2e5/final"
CAP_DROPOUT="$WORK_ROOT/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final"
CAP_NO_DROPOUT="$WORK_ROOT/checkpoints/mlm/mixed_case_no_dropout/capitalized_from_3class_steps3000_lr2e5/final"
CAP_NO_AUX="$WORK_ROOT/checkpoints/mlm/mixed_case_no_aux_loss/capitalized_from_3class_steps3000_lr2e5_drop01/final"

SEEDS=(13 21 34 55 89)
TASKS=(conll2003_ner kaggle_walia_ner)

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo=$(pwd)"
git rev-parse --short HEAD

require_checkpoint() {
  local checkpoint="$1"
  if [[ ! -f "$checkpoint/config.json" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
}

train_if_missing() {
  local final_dir="$1"
  local output_root="$2"
  local dropout="$3"
  local cap_loss_weight="$4"
  local log_name="$5"

  if [[ -f "$final_dir/config.json" ]]; then
    echo "SKIP pretraining: $final_dir already exists"
    return
  fi

  echo "RUN pretraining: output_root=$output_root dropout=$dropout cap_loss_weight=$cap_loss_weight"
  python -u scripts/run_mlm_pretraining.py \
    --model-kind capitalized \
    --initial-checkpoint "$CAP_3CLASS" \
    --corpus capitalization_real_acronym_mix \
    --output-root "$output_root" \
    --max-steps 3000 \
    --batch-size 32 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 \
    --capitalization-loss-weight "$cap_loss_weight" \
    --capitalization-class-weights 1,2,8,4 \
    --use-mixed-case-capitalization \
    --capitalization-embedding-dropout "$dropout" \
    2>&1 | tee "$LOG_ROOT/$log_name.log"
}

run_eval_variant() {
  local variant="$1"
  local checkpoint="$2"

  require_checkpoint "$checkpoint"
  echo "RUN eval variant=$variant checkpoint=$checkpoint"
  python -u scripts/resume_benchmark_sweep.py \
    --results-root "$ABLATION_ROOT/$variant" \
    --models capitalized_pretrained \
    --uncased-checkpoint "$UNC" \
    --cased-checkpoint "$CAS" \
    --capitalized-checkpoint "$checkpoint" \
    --seeds "${SEEDS[@]}" \
    --token-tasks "${TASKS[@]}" \
    --sequence-tasks \
    --token-epochs 3 \
    --token-batch-size 16 \
    --token-learning-rate 3e-5 \
    --no-save-model \
    2>&1 | tee "$LOG_ROOT/eval_${variant}.log"
}

require_checkpoint "$UNC"
require_checkpoint "$CAS"
require_checkpoint "$CAP_3CLASS"
require_checkpoint "$CAP_DROPOUT"

train_if_missing \
  "$CAP_NO_DROPOUT" \
  "$WORK_ROOT/checkpoints/mlm/mixed_case_no_dropout/capitalized_from_3class_steps3000_lr2e5" \
  "0.0" \
  "0.25" \
  "pretrain_mixed_case_no_dropout"

train_if_missing \
  "$CAP_NO_AUX" \
  "$WORK_ROOT/checkpoints/mlm/mixed_case_no_aux_loss/capitalized_from_3class_steps3000_lr2e5_drop01" \
  "0.1" \
  "0.0" \
  "pretrain_mixed_case_no_aux_loss"

run_eval_variant "three_class_existing" "$CAP_3CLASS"
run_eval_variant "four_class_dropout_current_best" "$CAP_DROPOUT"
run_eval_variant "four_class_no_dropout" "$CAP_NO_DROPOUT"
run_eval_variant "four_class_no_aux_loss" "$CAP_NO_AUX"

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
