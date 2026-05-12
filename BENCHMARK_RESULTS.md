# Benchmark Results

Results are appended here when a full benchmark run completes. Treat single-seed
wins as promising evidence, not final proof.

## CoNLL-2003 NER

Run date: 2026-05-10

Command:

```bash
python scripts/run_token_classification_benchmark.py \
  --benchmark conll2003_ner \
  --models uncased cased capitalized capitalized_pretrained \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/final \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --results-file /workspace/capitalization_embeddings/checkpoints/benchmarks/conll2003_ner/full_results.jsonl
```

Environment:

```text
RunPod RTX 4090
seed = 13
dataset = lhoestq/conll2003
metric = seqeval test F1
```

| Model | Continued MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.904548 | 0.898933 | 0.910234 | 0.980640 |
| `bert-base-cased` | no | 0.913257 | 0.905782 | 0.920857 | 0.982599 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.910126 | 0.900659 | 0.919795 | 0.982449 |
| `capitalized_pretrained` initialized from `bert-base-uncased` | yes | 0.913929 | 0.906762 | 0.921211 | 0.982642 |

Current read:

```text
capitalized_pretrained - bert-base-uncased = +0.009381 F1
capitalized_pretrained - bert-base-cased   = +0.000672 F1
```

This satisfies the primary goal on this run and narrowly satisfies the first
stretch goal on this run. Next replication target: repeat with multiple seeds
and add WNUT-17 NER.

## CoNLL-2003 NER With Matched Domain-Adaptive MLM

Run date: 2026-05-10

Pretraining command pattern:

```bash
python scripts/run_mlm_pretraining.py \
  --model-kind {uncased,cased,capitalized} \
  --corpus conll2003_train \
  --max-steps 1000 \
  --batch-size 32 \
  --gradient-accumulation-steps 2
```

Fine-tuning command:

```bash
python scripts/run_token_classification_benchmark.py \
  --benchmark conll2003_ner \
  --models uncased cased capitalized uncased_pretrained cased_pretrained capitalized_pretrained \
  --uncased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/conll2003_train/uncased_steps1000/final \
  --cased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/conll2003_train/cased_steps1000/final \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/conll2003_train/capitalized_steps1000/final \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --results-file /workspace/capitalization_embeddings/checkpoints/benchmarks/conll2003_ner/domain_pretrain_results.jsonl
```

| Model | Domain MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.904259 | 0.898880 | 0.909703 | 0.980575 |
| `bert-base-cased` | no | 0.913257 | 0.905782 | 0.920857 | 0.982599 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.910126 | 0.900659 | 0.919795 | 0.982449 |
| `uncased_pretrained` | yes | 0.903907 | 0.898530 | 0.909348 | 0.980467 |
| `cased_pretrained` | yes | 0.915311 | 0.909313 | 0.921388 | 0.982879 |
| `capitalized_pretrained` | yes | 0.916426 | 0.907906 | 0.925106 | 0.982685 |

Current read:

```text
capitalized_pretrained - uncased_pretrained = +0.012518 F1
capitalized_pretrained - cased_pretrained   = +0.001115 F1
```

This is a more useful comparison than the first result because all three model
families received matched MLM adaptation on the downstream domain before
fine-tuning.

## CoNLL-2003 NER With Matched Wikitext-103 MLM

Run date: 2026-05-10

Pretraining command pattern:

```bash
python scripts/run_mlm_pretraining.py \
  --model-kind {uncased,cased,capitalized} \
  --corpus wikitext103 \
  --max-steps 5000 \
  --batch-size 32 \
  --gradient-accumulation-steps 2
```

Notes:

- The `cased` Wikitext control was rerun cleanly in
  `cased_steps5000_clean` because resuming the partial checkpoint hit the
  PyTorch 2.4 / Transformers checkpoint-loading safety guard for optimizer
  state.
- The `capitalized` Wikitext MLM checkpoint reached capitalization accuracy
  `0.954554`, first-cap accuracy `0.878837`, all-caps accuracy `0.568528`,
  and token loss `1.560557` on the Wikitext validation subset.

Fine-tuning command:

```bash
python scripts/run_token_classification_benchmark.py \
  --benchmark conll2003_ner \
  --models uncased cased capitalized uncased_pretrained cased_pretrained capitalized_pretrained \
  --uncased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103/uncased_steps5000/final \
  --cased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103/cased_steps5000_clean/final \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103/capitalized_steps5000/final \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --results-file /workspace/capitalization_embeddings/checkpoints/benchmarks/conll2003_ner/wikitext_pretrain_results.jsonl
```

| Model | Wikitext MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.904259 | 0.898880 | 0.909703 | 0.980575 |
| `bert-base-cased` | no | 0.913257 | 0.905782 | 0.920857 | 0.982599 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.910126 | 0.900659 | 0.919795 | 0.982449 |
| `uncased_pretrained` | yes | 0.902818 | 0.895351 | 0.910411 | 0.980338 |
| `cased_pretrained` | yes | 0.910449 | 0.903856 | 0.917139 | 0.982492 |
| `capitalized_pretrained` | yes | 0.914140 | 0.907521 | 0.920857 | 0.982879 |

Current read:

```text
capitalized_pretrained - uncased_pretrained = +0.011322 F1
capitalized_pretrained - cased_pretrained   = +0.003691 F1
capitalized_pretrained - bert-base-cased    = +0.000883 F1
```

This directly controls for the possibility that the capitalized model only won
because it received extra generic MLM training. Under equal Wikitext-103
continued-pretraining budget, the capitalized model outperformed both
pretrained controls on this seed.

## CoNLL-2003 NER With Wikitext-103 Then Domain MLM

Run date: 2026-05-11

Sequential pretraining:

1. Continue MLM on Wikitext-103 for 5000 steps.
2. Continue MLM again on CoNLL-2003 train text for 1000 steps.
3. Fine-tune on CoNLL-2003 NER for 3 epochs.

This is the strictest CoNLL control so far: all model families received both
the generic MLM stage and the downstream-domain MLM stage.

Fine-tuning command:

```bash
python scripts/run_token_classification_benchmark.py \
  --benchmark conll2003_ner \
  --models uncased cased capitalized uncased_pretrained cased_pretrained capitalized_pretrained \
  --uncased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103_then_conll2003_train/uncased_steps1000/final \
  --cased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103_then_conll2003_train/cased_steps1000/final \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103_then_conll2003_train/capitalized_steps1000/final \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --results-file /workspace/capitalization_embeddings/checkpoints/benchmarks/conll2003_ner/wikitext_then_domain_pretrain_results.jsonl
```

| Model | Wikitext + domain MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.904355 | 0.898898 | 0.909880 | 0.980597 |
| `bert-base-cased` | no | 0.913257 | 0.905782 | 0.920857 | 0.982599 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.910126 | 0.900659 | 0.919795 | 0.982449 |
| `uncased_pretrained` | yes | 0.903192 | 0.894719 | 0.911827 | 0.980403 |
| `cased_pretrained` | yes | 0.915045 | 0.909998 | 0.920149 | 0.983224 |
| `capitalized_pretrained` | yes | 0.915385 | 0.908599 | 0.922273 | 0.982729 |

Current read:

```text
capitalized_pretrained - uncased_pretrained = +0.012193 F1
capitalized_pretrained - cased_pretrained   = +0.000340 F1
capitalized_pretrained - bert-base-cased    = +0.002128 F1
```

The margin over the matched cased control is small, but the direction remains
positive after controlling for both generic continued pretraining and
domain-adaptive pretraining.

## Task-Mix MLM Controls Across CoNLL, WNUT-17, OntoNotes, and PTB POS

Run date: 2026-05-11

Task-mix continued pretraining used Wikitext-103 checkpoints as the starting
point, then ran 3000 MLM steps on a capitalization-heavy mix of CoNLL-2003,
WNUT-17, OntoNotes v5, and PTB POS train text. The same budget was applied to
`bert-base-uncased`, `bert-base-cased`, and the capitalized-embedding model.

Pretraining checkpoints:

```text
uncased_pretrained:     /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103_then_task_mix/uncased_steps3000/final
cased_pretrained:       /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103_then_task_mix/cased_steps3000/final
capitalized_pretrained: /workspace/capitalization_embeddings/checkpoints/mlm/wikitext103_then_task_mix/capitalized_steps3000_clean/final
```

The capitalized task-mix MLM checkpoint reached capitalization accuracy
`0.937468`, first-cap accuracy `0.856048`, all-caps accuracy `0.468208`, and
none-case accuracy `0.977120` on the task-mix validation split.

### CoNLL-2003 NER

| Model | Task-mix MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.904548 | 0.898933 | 0.910234 | 0.980640 |
| `bert-base-cased` | no | 0.913257 | 0.905782 | 0.920857 | 0.982599 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.910126 | 0.900659 | 0.919795 | 0.982449 |
| `uncased_pretrained` | yes | 0.905051 | 0.898030 | 0.912181 | 0.980597 |
| `cased_pretrained` | yes | 0.914411 | 0.907711 | 0.921211 | 0.982815 |
| `capitalized_pretrained` | yes | 0.914753 | 0.909075 | 0.920503 | 0.982793 |

Current read:

```text
capitalized_pretrained - uncased_pretrained = +0.009703 F1
capitalized_pretrained - cased_pretrained   = +0.000342 F1
```

### WNUT-17 NER

| Model | Task-mix MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.461796 | 0.579832 | 0.383689 | 0.947287 |
| `bert-base-cased` | no | 0.460768 | 0.576602 | 0.383689 | 0.949085 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.428332 | 0.537815 | 0.355885 | 0.946390 |
| `uncased_pretrained` | yes | 0.452765 | 0.598174 | 0.364226 | 0.946646 |
| `cased_pretrained` | yes | 0.471038 | 0.573901 | 0.399444 | 0.947974 |
| `capitalized_pretrained` | yes | 0.448841 | 0.575362 | 0.367933 | 0.948313 |

Current read:

```text
capitalized_pretrained - uncased_pretrained = -0.003924 F1
capitalized_pretrained - cased_pretrained   = -0.022197 F1
```

WNUT-17 is currently a negative result for the method. This matters because it
tests noisier social-media entities rather than clean newswire capitalization.

### OntoNotes v5 NER

| Model | Task-mix MLM checkpoint | Test F1 | Test precision | Test recall | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | no | 0.871419 | 0.860808 | 0.882295 | 0.977600 |
| `bert-base-cased` | no | 0.886714 | 0.872421 | 0.901484 | 0.981810 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.888177 | 0.877362 | 0.899263 | 0.981365 |
| `uncased_pretrained` | yes | 0.873310 | 0.863230 | 0.883628 | 0.977796 |
| `cased_pretrained` | yes | 0.892229 | 0.880536 | 0.904237 | 0.982170 |
| `capitalized_pretrained` | yes | 0.888986 | 0.877673 | 0.900595 | 0.981758 |

Current read:

```text
capitalized - bert-base-uncased             = +0.016758 F1
capitalized - bert-base-cased               = +0.001463 F1
capitalized_pretrained - uncased_pretrained = +0.015676 F1
capitalized_pretrained - cased_pretrained   = -0.003243 F1
```

OntoNotes is the most encouraging raw-architecture result so far: the
capitalized-embedding model beats both raw uncased and raw cased on this seed.
Under the stricter matched task-mix MLM control, however, cased pretraining
wins by a clearer margin.

### PTB POS

| Model | Task-mix MLM checkpoint | Test accuracy | Test loss |
| --- | --- | ---: | ---: |
| `bert-base-uncased` | no | 0.973423 | 0.101505 |
| `bert-base-cased` | no | 0.976636 | 0.089682 |
| `capitalized` initialized from `bert-base-uncased` | no | 0.976979 | 0.092297 |
| `uncased_pretrained` | yes | 0.973054 | 0.103803 |
| `cased_pretrained` | yes | 0.977532 | 0.085746 |
| `capitalized_pretrained` | yes | 0.977084 | 0.092062 |

Current read:

```text
capitalized_pretrained - uncased_pretrained = +0.004030 accuracy
capitalized_pretrained - cased_pretrained   = -0.000448 accuracy
```

PTB is close, but the matched cased control is still ahead after task-mix MLM.
The raw capitalized model does slightly beat raw cased on this seed.

## Augmented Task-Mix Capitalized MLM

Run date: 2026-05-11

This experiment targeted the weak all-caps channel from the first task-mix MLM
checkpoint. It continued from:

```text
/workspace/capitalization_embeddings/checkpoints/mlm/wikitext103_then_task_mix/capitalized_steps3000_clean/final
```

Training changes:

```text
corpus: capitalization_task_mix_augmented
max_steps: 10000
capitalization_loss_weight: 0.5
capitalization_class_weights: [1.0, 2.0, 8.0]
checkpoint: /workspace/capitalization_embeddings/checkpoints/mlm/task_mix_augmented/capitalized_from_task_mix_steps10000_wcap/final
```

Capitalization diagnostics:

| Metric | Previous task-mix checkpoint | Augmented weighted checkpoint |
| --- | ---: | ---: |
| capitalization accuracy | 0.937468 | 0.915205 |
| none accuracy | 0.977120 | 0.937809 |
| first-cap accuracy | 0.856048 | 0.825383 |
| all-caps accuracy | 0.468208 | 0.940860 |

The all-caps channel improved dramatically, but the weighting was aggressive and
reduced no-cap/first-cap accuracy.

Downstream check:

| Benchmark | Previous task-mix capitalized | Augmented weighted capitalized | Prior matched cased |
| --- | ---: | ---: | ---: |
| CoNLL-2003 NER F1 | 0.914753 | 0.917202 | 0.914411 |
| OntoNotes v5 NER F1 | 0.888986 | 0.886542 | 0.892229 |

Current read:

```text
CoNLL improved enough to become the strongest single-seed controlled result so far.
OntoNotes worsened, suggesting the all-caps weighting overcorrected and hurt broader
case behavior.
```

Follow-up run: same augmented corpus with softer capitalization weighting
(`capitalization_loss_weight=0.35`, class weights `[1.0, 1.5, 4.0]`).

Capitalization diagnostics:

| Metric | Soft augmented weighted checkpoint |
| --- | ---: |
| capitalization accuracy | 0.924149 |
| none accuracy | 0.951953 |
| first-cap accuracy | 0.826235 |
| all-caps accuracy | 0.876344 |
| eval token loss | 2.133818 |

Downstream check:

| Benchmark | Soft augmented weighted capitalized |
| --- | ---: |
| CoNLL-2003 NER F1 | 0.915094 |
| OntoNotes v5 NER F1 | 0.886473 |

Softening the weights recovered some no-cap accuracy but did not recover
OntoNotes, and it gave up most of the aggressive run's CoNLL gain. The next
hypothesis is that synthetic capitalization augmentation is causing distribution
shift. The next run should use the natural task-mix corpus with weighted
capitalization loss but without synthetic case variants.

Natural task-mix follow-up: same soft capitalization weights, but without
synthetic case augmentation.

```text
corpus: capitalization_task_mix
max_steps: 10000
capitalization_loss_weight: 0.35
capitalization_class_weights: [1.0, 1.5, 4.0]
checkpoint: /workspace/capitalization_embeddings/checkpoints/mlm/task_mix_weighted/capitalized_from_task_mix_steps10000_softcap/final
```

Capitalization diagnostics:

| Metric | Natural weighted checkpoint |
| --- | ---: |
| capitalization accuracy | 0.950769 |
| none accuracy | 0.971038 |
| first-cap accuracy | 0.885391 |
| all-caps accuracy | 0.903030 |
| eval token loss | 2.162145 |

Downstream check:

| Benchmark | Natural weighted capitalized |
| --- | ---: |
| CoNLL-2003 NER F1 | 0.914100 |
| OntoNotes v5 NER F1 | 0.887110 |

This run gives the best capitalization diagnostic balance, but downstream
performance does not improve. The current evidence suggests that directly
optimizing auxiliary capitalization accuracy is not sufficient; the next data
direction should be real acronym-heavy corpora rather than synthetic case
variants or heavier capitalization loss.

## Real Acronym Corpus Follow-Up

Run date: 2026-05-11

Added `capitalization_real_acronym_mix`, a natural-text corpus that extends the
task mix with acronym-heavy rows from:

```text
ccdv/pubmed-summarization: biomedical abstracts/articles
billsum: government/legal bill text
lex_glue/scotus: legal opinions
```

Rows are chunked and filtered/ranked by uppercase acronym signal instead of
synthetically changing word case. The first run keeps the standard
capitalization objective:

```text
corpus: capitalization_real_acronym_mix
max_steps: 10000
capitalization_loss_weight: 0.25
capitalization_class_weights: none
checkpoint: /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/capitalized_from_task_mix_steps10000/final
```

Hypothesis: real acronym-heavy text should improve all-caps behavior without the
OntoNotes regression caused by synthetic capitalization augmentation.

### 10k-Step Result

The 10k real-acronym continuation improved the auxiliary capitalization channel
substantially, especially all-caps recognition, but it still over-adapted for
broad NER.

```text
eval_capitalization_accuracy: 0.944736
eval_capitalization_none_accuracy: 0.973789
eval_capitalization_first_cap_accuracy: 0.868936
eval_capitalization_all_caps_accuracy: 0.711957
eval_capitalization_all_caps_count: 184
```

| Benchmark | F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| CoNLL-2003 NER | 0.915201 | 0.909441 | 0.921034 | 0.982513 |
| OntoNotes v5 NER | 0.883764 | 0.879065 | 0.888514 | 0.980985 |

Compared with the previous task-mix checkpoint, CoNLL moved up from `0.914753`
to `0.915201`, while OntoNotes fell from `0.888986` to `0.883764`. This suggests
the corpus direction is useful for all-caps learning, but 10k steps at `5e-5`
is too much continuation from the stronger task-mix checkpoint.

Follow-up started:

```text
corpus: capitalization_real_acronym_mix
max_steps: 3000
learning_rate: 2e-5
capitalization_loss_weight: 0.25
checkpoint: /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/capitalized_from_task_mix_steps3000_lr2e5/final
```

The 3k/lower-LR run completed with a smaller capitalization shift:

```text
eval_capitalization_accuracy: 0.936093
eval_capitalization_none_accuracy: 0.975703
eval_capitalization_first_cap_accuracy: 0.851754
eval_capitalization_all_caps_accuracy: 0.513228
eval_capitalization_all_caps_count: 189
```

This is a modest all-caps improvement over the task-mix checkpoint, not the
large correction seen in the 10k run.

| Benchmark | F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| CoNLL-2003 NER | 0.915129 | 0.908266 | 0.922096 | 0.982987 |
| OntoNotes v5 NER | 0.888013 | 0.876113 | 0.900240 | 0.981738 |
| WNUT-17 NER | 0.485396 | 0.568408 | 0.423540 | 0.950451 |
| PTB POS | n/a | n/a | n/a | 0.977269 |

Compared with the task-mix checkpoint, CoNLL improved from `0.914753` to
`0.915129`, while OntoNotes moved from `0.888986` to `0.888013`. The gentler
continuation preserved most of the broad NER signal but did not improve it. The
largest gain was WNUT-17, where F1 improved from `0.448841` to `0.485396`.
PTB POS also moved slightly from `0.977084` to `0.977269`.

Matched cased/uncased real-acronym continuation is the next control. This will
test whether the WNUT gain comes from the capitalization architecture or from
extra acronym-heavy MLM exposure alone.

Matched-control pretraining completed:

```text
uncased checkpoint: /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final
uncased eval_loss: 1.938038
cased checkpoint: /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final
cased eval_loss: 1.912665
```

Four-task downstream evaluation completed for `uncased_pretrained` and
`cased_pretrained`.

| Benchmark | Metric | Capitalized | Matched uncased | Matched cased |
| --- | --- | ---: | ---: | ---: |
| CoNLL-2003 NER | F1 | 0.915129 | 0.903963 | 0.914085 |
| WNUT-17 NER | F1 | 0.485396 | 0.449095 | 0.461375 |
| OntoNotes v5 NER | F1 | 0.888013 | 0.873388 | 0.889647 |
| PTB POS | Accuracy | 0.977269 | 0.973634 | 0.977453 |

The CoNLL and WNUT gains survive the matched extra-MLM control. WNUT is the
clearest architecture-positive result so far: the capitalization-embedding model
beats both cased and uncased models after all three receive the same
real-acronym continuation. OntoNotes and PTB still favor matched cased by small
margins, but the capitalized model remains well above matched uncased.
