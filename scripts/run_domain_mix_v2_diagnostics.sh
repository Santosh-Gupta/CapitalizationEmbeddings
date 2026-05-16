#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/repos/CapitalizationEmbeddings}"
WORK_ROOT="${CAP_EMB_WORK_ROOT:-/workspace/capitalization_embeddings}"
PYTHON="${PYTHON:-python}"

cd "$REPO_ROOT"
export CAP_EMB_WORK_ROOT="$WORK_ROOT"

V2_ROOT="$WORK_ROOT/checkpoints/mlm/domain_mix_v2"
UNC="$V2_ROOT/uncased_from_round2_steps3000_lr2e5/final"
CAS="$V2_ROOT/cased_from_round2_steps3000_lr2e5/final"
CAP="$V2_ROOT/capitalized_from_mixed_case_current_steps3000_lr2e5/final"
RESULTS="$WORK_ROOT/checkpoints/domain_mix_v2_diagnostics_3seed"

"$PYTHON" -u scripts/resume_benchmark_sweep.py \
  --results-root "$RESULTS" \
  --models uncased_pretrained cased_pretrained capitalized_pretrained \
  --uncased-checkpoint "$UNC" \
  --cased-checkpoint "$CAS" \
  --capitalized-checkpoint "$CAP" \
  --seeds 13 21 34 \
  --token-tasks conll2003_ner wnut17_ner \
  --sequence-tasks tweet_eval_emoji trec_fine scientific_relations_combined tweet_eval_irony twenty_newsgroups \
  --token-epochs 3 \
  --token-batch-size 16 \
  --token-learning-rate 3e-5 \
  --sequence-epochs 3 \
  --sequence-batch-size 16 \
  --sequence-learning-rate 2e-5 \
  --no-save-model

echo "diagnostic results=$RESULTS"
