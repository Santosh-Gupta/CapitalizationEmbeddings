#!/usr/bin/env bash
set -euo pipefail

# Cost-aware rerun of the case-channel ablation sweep.
#
# The original launcher used token batch size 16, but the RTX 4090 only used
# about 3 GB VRAM on Walia NER. This launcher writes to a separate result root
# and uses batch size 64 so every row in this rerun remains internally
# comparable and the old batch-16 rows are not mixed into the same table.

REPO_ROOT="${CAP_EMB_REPO:-/workspace/repos/CapitalizationEmbeddings}"
WORK_ROOT="${CAP_EMB_WORK_ROOT:-/workspace/capitalization_embeddings}"

cd "$REPO_ROOT"

LOG_ROOT="$WORK_ROOT/logs/ablations_case_channel_b64"
ABLATION_ROOT="$WORK_ROOT/checkpoints/ablations_case_channel_5seed_b64"
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
echo "token_batch_size=64"
echo "result_root=$ABLATION_ROOT"

require_checkpoint() {
  local checkpoint="$1"
  if [[ ! -f "$checkpoint/config.json" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
}

for checkpoint in "$UNC" "$CAS" "$CAP_3CLASS" "$CAP_DROPOUT" "$CAP_NO_DROPOUT" "$CAP_NO_AUX"; do
  require_checkpoint "$checkpoint"
done

run_eval_variant() {
  local variant="$1"
  local checkpoint="$2"

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
    --token-batch-size 64 \
    --token-learning-rate 3e-5 \
    --no-save-model \
    2>&1 | tee "$LOG_ROOT/eval_${variant}.log"
}

run_eval_variant "three_class_existing" "$CAP_3CLASS"
run_eval_variant "four_class_dropout_current_best" "$CAP_DROPOUT"
run_eval_variant "four_class_no_dropout" "$CAP_NO_DROPOUT"
run_eval_variant "four_class_no_aux_loss" "$CAP_NO_AUX"

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
