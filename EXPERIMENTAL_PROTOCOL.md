# Experimental Protocol

This document locks the experiment design for paper writing and future GPU
runs. Changes after this point should be recorded as protocol amendments.

## Model Families

All headline comparisons use three matched model families:

| Model key | Base model | Continued pretraining checkpoint | Role |
| --- | --- | --- | --- |
| `uncased_pretrained` | `bert-base-uncased` | `/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final` | lexical-sharing baseline |
| `cased_pretrained` | `bert-base-cased` | `/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final` | conventional cased baseline |
| `capitalized_pretrained` | `bert-base-uncased` + capitalization embeddings | `/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final` | proposed method |

The capitalized model uses four case states:

```text
0 = no capitalization feature / lowercase / special token
1 = first-cap
2 = all-caps
3 = mixed-case
```

## Continued Pretraining

The current best capitalized checkpoint uses:

```text
corpus: capitalization_real_acronym_mix
max_steps: 3000
learning_rate: 2e-5
capitalization_loss_weight: 0.25
capitalization_class_weights: [1, 2, 8, 4]
capitalization_embedding_dropout: 0.1
```

Rows 0-2 of the capitalization embedding and auxiliary classifier tables are
restored from the prior three-class checkpoint when expanding to the mixed-case
fourth class.

Matched cased and uncased controls receive the same real-acronym continuation
recipe without the capitalization architecture.

## Downstream Fine-Tuning

Default downstream settings:

| Task family | Epochs | Batch size | Learning rate |
| --- | ---: | ---: | ---: |
| Token classification | 3 | 16 | 3e-5 |
| Sequence classification | 3 | 16 | 2e-5 |

Default seed set for 5-seed diagnostics:

```text
13 21 34 55 89
```

Additional 15 seeds used for 20-seed headline expansions:

```text
144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393
```

Every completed benchmark must save:

- one JSONL row per model/seed;
- per-example predictions for paired bootstrap;
- enough metadata to recover model key, seed, split, checkpoint, and metric.

## Benchmark Tiers

Primary token/entity capitalization benchmarks:

| Benchmark | Metric | Current seed status | Paper role |
| --- | --- | ---: | --- |
| CoNLL-2003 NER | entity F1 | 20 | primary |
| WNUT-17 NER | entity F1 | 20 | primary, high variance |
| OntoNotes v5 NER | entity F1 | 5 | primary/supporting until expanded |
| PTB POS | accuracy | 5 | primary/supporting until expanded |
| Kaggle/Walia NER | entity F1 | 5 | supporting, because original reported setup used BERT embeddings inside another architecture |

Uncased-favored/general controls:

| Benchmark | Metric | Current seed status | Paper role |
| --- | --- | ---: | --- |
| TweetEval Irony | macro-F1 | 20 | control |
| TweetEval Offensive | macro-F1 | 20 | control |
| SST-5 | accuracy | 20 | control |
| 20 Newsgroups | accuracy | 20 | control |

Negative controls and appendix diagnostics:

| Benchmark | Current read | Paper role |
| --- | --- | --- |
| TweetEval Emoji | cased dominates; cap near uncased | negative control |
| TREC Fine | cap underperforms both | negative control |
| SciERC/combined scientific relations | uncased dominates | appendix limitation |
| SciEntsBank | mixed, often unfavorable | appendix limitation |
| iSarcasmEval EN | cap ties both with small negative mean | neutral appendix |

## Statistical Tests

Use two complementary uncertainty views:

1. Seed-level replication:
   - mean, standard deviation, paired seed deltas;
   - superiority tests for "beats" claims;
   - non-inferiority tests for "matches the best baseline" claims.
2. Example-level paired bootstrap:
   - paired on identical test examples and saved predictions;
   - Holm-Bonferroni correction within each benchmark family.

Predeclared practical margins:

| Margin | Use |
| ---: | --- |
| 0.002 | strict headline non-inferiority |
| 0.005 | practical appendix/supporting non-inferiority |

Do not write that a non-significant difference proves equality. Use
non-inferiority language for "matches" claims.

## Model Selection Rule

The paper should use one current-best capitalization checkpoint selected before
the final benchmark table:

```text
/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final
```

Do not select different capitalization checkpoints per downstream benchmark in
the main table. Per-task checkpoint selection can appear only as exploratory
analysis or appendix diagnostics.

## Compute Gate

RunPod should stay stopped until:

1. the ablation matrix is finalized;
2. exact commands are added to `RUN_LEDGER.md` before launch;
3. the expected result root and cleanup plan are specified;
4. the user explicitly restarts the pod.
