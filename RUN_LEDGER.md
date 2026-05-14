# Run Ledger

This file records launched GPU/CPU experiment batches so work is not duplicated
after context compression.

## RunPod

```text
host: root@203.57.40.173 -p 10267
repo: /workspace/repos/CapitalizationEmbeddings
checkpoint root: /workspace/capitalization_embeddings/checkpoints
log root: /workspace/capitalization_embeddings/logs
reports root: /workspace/capitalization_embeddings/reports
```

Before launching work:

1. Confirm GPU/process state with `nvidia-smi` and `pgrep`.
2. Add or update an entry in this file with command, seeds, roots, and log path.
3. Use `resume_benchmark_sweep.py` so completed model/seed rows are skipped.
4. After completion, update this file and `PAPER_EVIDENCE_STATUS.md`.

## Completed Batches

### Mixed-Case Required Token 5-Seed Completion

Status: completed.

Result roots:

```text
capitalized current-best:
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed

matched cased/uncased OntoNotes/PTB:
/workspace/capitalization_embeddings/checkpoints/required_token_baselines_3seed
```

Seeds:

```text
13 21 34 55 89
```

Tasks:

```text
conll2003_ner
wnut17_ner
ontonotes5_ner
ptb_pos
```

Notes:

- Historical root names include `3seed`, but these roots now contain 5 seeds for
  the rows described above.
- CoNLL/WNUT cased and uncased controls are in
  `/workspace/capitalization_embeddings/checkpoints/significance_5seed`.

### Mixed-Case Selected Sequence 5-Seed Completion

Status: completed.

Result roots:

```text
capitalized current-best:
/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed

matched cased/uncased controls:
/workspace/capitalization_embeddings/checkpoints/significance_5seed
```

Seeds:

```text
13 21 34 55 89
```

Tasks:

```text
tweet_eval_irony
tweet_eval_offensive
sst5
twenty_newsgroups
```

## Active Or Next Batch

### Paper Final High-Variance 20-Seed Expansion

Status: stopped early; partial results retained.

Started: 2026-05-14 UTC.

RunPod PID:

```text
60740 (exited)
```

Purpose:

- Expand WNUT-17 and CoNLL-2003 from 5 to 20 seeds for all three matched model
  families.
- Expand selected uncased-favored sequence tasks from 5 to 20 seeds for all
  three matched model families.
- This is the main remaining GPU-heavy batch before deciding whether the claim
  is statistically strong enough or must be narrowed.

Additional seeds:

```text
144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393
```

Checkpoints:

```text
capitalized:
/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final

uncased:
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final

cased:
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final
```

Result roots:

```text
capitalized token:
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed

capitalized sequence:
/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed

cased/uncased token and sequence:
/workspace/capitalization_embeddings/checkpoints/significance_5seed
```

Launcher/log:

```text
/workspace/capitalization_embeddings/logs/run_paper_final_20seed_expansion.sh
/workspace/capitalization_embeddings/logs/paper_final_20seed_expansion.log
```

Tasks:

```text
token:
conll2003_ner
wnut17_ner

sequence:
tweet_eval_irony
tweet_eval_offensive
sst5
twenty_newsgroups
```

Partial completion observed on 2026-05-14 UTC:

```text
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed
  conll2003_ner capitalized_pretrained: 20 seeds
  wnut17_ner capitalized_pretrained: 20 seeds

/workspace/capitalization_embeddings/checkpoints/significance_5seed
  conll2003_ner uncased_pretrained: 20 seeds
  conll2003_ner cased_pretrained: 20 seeds
  wnut17_ner uncased_pretrained: 13 seeds
  wnut17_ner cased_pretrained: 13 seeds
  selected sequence tasks: 5 seeds/model

/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed
  selected sequence tasks capitalized_pretrained: 5 seeds
```

### Paper Final Missing 20-Seed Remainder

Status: queued for immediate RunPod launch.

Purpose:

- Fill only rows missing from the previous expansion, without rerunning completed
  CoNLL or current-best WNUT rows.
- Complete WNUT-17 cased/uncased controls to 20 seeds.
- Complete the selected sequence task current-best capitalized rows to 20 seeds.
- Complete the selected sequence task cased/uncased controls to 20 seeds.

Launcher/log:

```text
/workspace/repos/CapitalizationEmbeddings/scripts/run_paper_final_missing_20seed_remainder.sh
/workspace/capitalization_embeddings/logs/paper_final_missing_20seed_remainder.log
```

Result roots:

```text
/workspace/capitalization_embeddings/checkpoints/significance_5seed
/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed
```
