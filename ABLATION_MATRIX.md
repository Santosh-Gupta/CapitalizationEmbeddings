# Ablation Matrix

This matrix defines the GPU work that is worth doing after the non-GPU paper
artifacts are complete. It is intentionally narrow: each ablation should answer
a reviewer-facing question.

## Core Questions

| Question | Ablation needed | Why it matters |
| --- | --- | --- |
| Is the gain from the architecture rather than extra MLM exposure? | matched cased/uncased continuation controls | already done for main checkpoints |
| Is any case feature enough, or does mixed-case matter? | 3-state vs 4-state capitalization embeddings | tests whether `iPhone`/`eBay`-style forms need a separate state |
| Does regularizing the case channel help? | 4-state with vs without capitalization embedding dropout | tests whether the model can ignore noisy case when needed |
| Is the auxiliary capitalization head necessary after pretraining? | pretrain with vs without capitalization prediction loss | separates learned case embedding from auxiliary objective |
| Does the method only help token/entity tasks? | token-task ablations plus negative-control sequence tasks | supports the narrowed paper claim |

## Existing Checkpoint Families

| Family | Checkpoint/status | Use |
| --- | --- | --- |
| 3-state task-mix/real-acronym | existing historical checkpoints in `BENCHMARK_RESULTS.md` | exploratory context; not current-best |
| 4-state mixed-case + dropout | `/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final` | current-best main method |
| matched real-acronym uncased | `/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final` | matched baseline |
| matched real-acronym cased | `/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final` | matched baseline |

## Next GPU Ablations

Run these only after commands and result roots are added to `RUN_LEDGER.md`.

| Priority | Ablation | Pretraining needed? | Downstream tasks | Seeds | Decision rule |
| ---: | --- | --- | --- | ---: | --- |
| 1 | 4-state mixed-case without capitalization embedding dropout | yes | CoNLL, Walia, SST-5, 20NG | 5 | If worse than dropout on token tasks or uncased controls, keep dropout |
| 2 | 3-state first/all only under the same real-acronym recipe | maybe reuse existing if checkpoint is clean; otherwise yes | CoNLL, Walia, OntoNotes, PTB | 5 | Tests whether mixed-case state is necessary |
| 3 | 4-state with no auxiliary capitalization loss | yes | CoNLL, Walia | 5 | Tests whether the case channel learns without direct supervision |
| 4 | no-capitalization-embedding control initialized from uncased continuation | no new pretraining if using matched uncased | CoNLL, Walia, OntoNotes, PTB | 5 | Confirms gains are not just runner/tokenization artifacts |
| 5 | expand Walia from 5 to 20 seeds for current-best method and controls | no | Walia | +15 | Promotes Walia from supporting to stronger main-table evidence |
| 6 | expand OntoNotes/PTB to 20 seeds | no | OntoNotes, PTB | +15 | Only if the paper needs more token-task replication |

## Recommended First GPU Batch

The first resumed GPU session should be small and decisive:

```text
1. Train or locate 4-state no-dropout checkpoint.
2. Run 5-seed CoNLL and Walia for:
   - current-best 4-state + dropout
   - 4-state no-dropout
   - matched uncased
   - matched cased
3. Stop and inspect before expanding.
```

If no-dropout is not better, do not spend more compute on it. Move to 3-state
versus 4-state and no-auxiliary-loss ablations.

Prepared launcher:

```bash
cd /workspace/repos/CapitalizationEmbeddings
bash scripts/run_case_ablation_batch.sh
```

The launcher is idempotent. It checks for existing ablation checkpoints, trains
missing 4-state no-dropout and 4-state no-aux-loss checkpoints, then runs
5-seed CoNLL/Walia evaluations into:

```text
/workspace/capitalization_embeddings/checkpoints/ablations_case_channel_5seed
```

## What Not To Run First

Do not prioritize:

- 20+ seeds on TweetEval Emoji or TREC Fine;
- more scientific relation or SciEntsBank sweeps;
- Yahoo Answers or other large sequence tasks;
- per-task checkpoint hunting.

Those do not strengthen the narrowed token/entity capitalization claim enough
to justify the GPU cost at this stage.
