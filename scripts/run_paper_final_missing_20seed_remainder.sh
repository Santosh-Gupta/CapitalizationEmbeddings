#!/usr/bin/env bash
set -euo pipefail

cd /workspace/repos/CapitalizationEmbeddings

UNC="/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final"
CAS="/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final"
CAP="/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final"

SEEDS=(144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393)

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo=$(pwd)"
git rev-parse --short HEAD

# The earlier 20-seed expansion completed CoNLL controls and current-best
# capitalized CoNLL/WNUT, then stopped with WNUT controls at 13/20.
python -u scripts/resume_benchmark_sweep.py \
  --results-root /workspace/capitalization_embeddings/checkpoints/significance_5seed \
  --models uncased_pretrained cased_pretrained \
  --uncased-checkpoint "$UNC" \
  --cased-checkpoint "$CAS" \
  --capitalized-checkpoint "$CAP" \
  --seeds "${SEEDS[@]}" \
  --token-tasks wnut17_ner \
  --sequence-tasks \
  --token-epochs 3 \
  --token-batch-size 16 \
  --token-learning-rate 3e-5 \
  --no-save-model

# Expand current-best capitalized results on the four selected
# uncased-favored sequence tasks from 5 to 20 seeds.
python -u scripts/resume_benchmark_sweep.py \
  --results-root /workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed \
  --models capitalized_pretrained \
  --uncased-checkpoint "$UNC" \
  --cased-checkpoint "$CAS" \
  --capitalized-checkpoint "$CAP" \
  --seeds "${SEEDS[@]}" \
  --token-tasks \
  --sequence-tasks tweet_eval_irony tweet_eval_offensive sst5 twenty_newsgroups \
  --sequence-epochs 3 \
  --sequence-batch-size 16 \
  --sequence-learning-rate 2e-5 \
  --no-save-model

# Expand the matched cased and uncased controls for the same sequence tasks.
python -u scripts/resume_benchmark_sweep.py \
  --results-root /workspace/capitalization_embeddings/checkpoints/significance_5seed \
  --models uncased_pretrained cased_pretrained \
  --uncased-checkpoint "$UNC" \
  --cased-checkpoint "$CAS" \
  --capitalized-checkpoint "$CAP" \
  --seeds "${SEEDS[@]}" \
  --token-tasks \
  --sequence-tasks tweet_eval_irony tweet_eval_offensive sst5 twenty_newsgroups \
  --sequence-epochs 3 \
  --sequence-batch-size 16 \
  --sequence-learning-rate 2e-5 \
  --no-save-model

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
