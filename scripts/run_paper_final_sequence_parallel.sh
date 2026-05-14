#!/usr/bin/env bash
set -euo pipefail

cd /workspace/repos/CapitalizationEmbeddings

UNC="/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final"
CAS="/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final"
CAP="/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final"

SEEDS=(144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393)
TASKS=(tweet_eval_irony tweet_eval_offensive sst5 twenty_newsgroups)

run_task() {
  local task="$1"
  local log_dir="/workspace/capitalization_embeddings/logs/sequence_parallel"
  mkdir -p "$log_dir"
  {
    echo "START task=$task $(date -u +%Y-%m-%dT%H:%M:%SZ)"

    python -u scripts/resume_benchmark_sweep.py \
      --results-root /workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed \
      --models capitalized_pretrained \
      --uncased-checkpoint "$UNC" \
      --cased-checkpoint "$CAS" \
      --capitalized-checkpoint "$CAP" \
      --seeds "${SEEDS[@]}" \
      --token-tasks \
      --sequence-tasks "$task" \
      --sequence-epochs 3 \
      --sequence-batch-size 16 \
      --sequence-learning-rate 2e-5 \
      --no-save-model

    python -u scripts/resume_benchmark_sweep.py \
      --results-root /workspace/capitalization_embeddings/checkpoints/significance_5seed \
      --models uncased_pretrained cased_pretrained \
      --uncased-checkpoint "$UNC" \
      --cased-checkpoint "$CAS" \
      --capitalized-checkpoint "$CAP" \
      --seeds "${SEEDS[@]}" \
      --token-tasks \
      --sequence-tasks "$task" \
      --sequence-epochs 3 \
      --sequence-batch-size 16 \
      --sequence-learning-rate 2e-5 \
      --no-save-model

    echo "DONE task=$task $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$log_dir/${task}.log" 2>&1
}

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo=$(pwd)"
git rev-parse --short HEAD

pids=()
for task in "${TASKS[@]}"; do
  run_task "$task" &
  pids+=("$!")
  echo "launched task=$task pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
