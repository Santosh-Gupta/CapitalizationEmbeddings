# Case-Channel Ablation, Batch-64, 5 Seeds

Run date: 2026-05-16

Remote result root:

```text
/workspace/capitalization_embeddings/checkpoints/ablations_case_channel_5seed_b64
```

Launcher:

```bash
cd /workspace/repos/CapitalizationEmbeddings
bash scripts/run_case_ablation_batch_b64.sh
```

Protocol:

```text
Models: capitalized_pretrained only
Variants:
  - three_class_existing
  - four_class_dropout_current_best
  - four_class_no_dropout
  - four_class_no_aux_loss
Tasks: conll2003_ner, kaggle_walia_ner
Seeds: 13, 21, 34, 55, 89
Epochs: 3
Token batch size: 64
Learning rate: 3e-5
Save models: no
```

The batch-64 rerun was created because the first batch-16 ablation run used
only about 3 GB VRAM on Walia NER and was projected to spend many extra
GPU-hours. The batch-16 partial rows were preserved remotely under
`ablations_case_channel_5seed`, but this report only uses the internally
consistent batch-64 root.

## Mean Test F1

| Variant | CoNLL-2003 NER | Kaggle/Walia NER |
| --- | ---: | ---: |
| `four_class_dropout_current_best` | 0.911972 +/- 0.001666 | 0.836846 +/- 0.005355 |
| `four_class_no_aux_loss` | 0.911216 +/- 0.003689 | 0.836203 +/- 0.005220 |
| `four_class_no_dropout` | 0.910332 +/- 0.002034 | 0.836242 +/- 0.006351 |
| `three_class_existing` | 0.909972 +/- 0.001788 | 0.835011 +/- 0.006435 |

## Seed Values

### CoNLL-2003 NER

| Variant | Seed F1 values |
| --- | --- |
| `four_class_dropout_current_best` | 0.911352, 0.911623, 0.914636, 0.912136, 0.910110 |
| `four_class_no_aux_loss` | 0.913299, 0.911018, 0.916081, 0.909186, 0.906496 |
| `four_class_no_dropout` | 0.907960, 0.910079, 0.913066, 0.908979, 0.911577 |
| `three_class_existing` | 0.908406, 0.908645, 0.912887, 0.910160, 0.909761 |

### Kaggle/Walia NER

| Variant | Seed F1 values |
| --- | --- |
| `four_class_dropout_current_best` | 0.833842, 0.838302, 0.829087, 0.841349, 0.841649 |
| `four_class_no_aux_loss` | 0.833094, 0.836062, 0.829301, 0.841044, 0.841515 |
| `four_class_no_dropout` | 0.830693, 0.838165, 0.828476, 0.841025, 0.842850 |
| `three_class_existing` | 0.826630, 0.838561, 0.829606, 0.840907, 0.839350 |

## Immediate Read

The best mean on both token/entity tasks is the current-best 4-state model with
capitalization embedding dropout. This supports keeping the dropout variant as
the default method.

The auxiliary capitalization loss does not look critical in this 5-seed
ablation: `four_class_no_aux_loss` is close to the dropout model on both tasks.
That is useful for the paper discussion, but not strong enough by itself to
drop the auxiliary objective.

The 4-state no-dropout and 3-state variants trail the dropout model on both
tasks. The differences are small, so this should be treated as ablation
evidence, not a standalone statistical claim.
