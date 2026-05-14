# Paper Evidence Status

Last updated: 2026-05-14

This file is the durable project ledger for paper readiness. It summarizes what
is supported by completed runs, what is still preliminary, and what must be run
before making top-tier conference or journal claims.

## Current Answer

We do not yet have an iron-clad paper package. We do have a strong and
publishable-looking core result:

```text
Capitalization embeddings recover most of the benefit of cased BERT on
capitalization-sensitive token tasks while keeping the lexical sharing of an
uncased vocabulary.
```

The current evidence is strongest for cased-favored token tasks. The mixed-case
capitalization variant beats matched uncased controls on all four required token
benchmarks. Against matched cased controls, it wins by mean on CoNLL-2003,
WNUT-17, and PTB POS, and is effectively tied on OntoNotes v5.

The current evidence is not strong enough for a broad "strictly better than
cased and uncased BERT" claim. Some uncased-favored and cased-favored sequence
benchmarks are misses.

## Current Best Method

Use the mixed-case/dropout capitalization model as the current best
capitalized-embedding variant:

```text
checkpoint:
/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final

starting checkpoint:
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/capitalized_from_task_mix_steps3000_lr2e5/final

training:
corpus = capitalization_real_acronym_mix
max_steps = 3000
learning_rate = 2e-5
capitalization_loss_weight = 0.25
capitalization_class_weights = [1, 2, 8, 4]
capitalization_embedding_dropout = 0.1
capitalization classes = none, first-cap, all-caps, mixed-case
```

Important implementation detail: when growing from the 3-class capitalization
embedding/table to the 4-class mixed-case table, rows 0-2 are restored from the
previous checkpoint and only the new mixed-case row is newly initialized.

## Source Result Roots

```text
5-seed 3-class core:
/workspace/capitalization_embeddings/checkpoints/significance_5seed

5-seed sequence/cased-favored diagnostics:
/workspace/capitalization_embeddings/checkpoints/cased_sequence_5seed

5-seed scientific relation diagnostics:
/workspace/capitalization_embeddings/checkpoints/scientific_5seed
/workspace/capitalization_embeddings/checkpoints/semeval2018_validation_5seed

5-seed SciEntsBank diagnostics:
/workspace/capitalization_embeddings/checkpoints/scientbank_5seed

Mixed-case token task run, now expanded to 5 seeds despite the historical path
name:
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed

Matched cased/uncased controls for OntoNotes/PTB, now expanded to 5 seeds
despite the historical path name:
/workspace/capitalization_embeddings/checkpoints/required_token_baselines_3seed

Mixed-case sequence run for current-best method:
/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed

Bootstrap/statistical reports:
/workspace/capitalization_embeddings/reports
```

Regenerate an evidence ledger from the current JSONL files with:

```bash
python scripts/collect_evidence_status.py \
  --checkpoint-root /workspace/capitalization_embeddings/checkpoints
```

Do not report `/workspace/capitalization_embeddings/checkpoints/scientific_5seed/semeval2018_task7`
as a final benchmark result; that run produced all-zero values. Use
`semeval2018_validation_5seed/semeval2018_task7` for the corrected validation
run.

## Required Token Benchmarks

These four should remain in the paper. They are the cleanest direct test of
whether capitalization embeddings recover cased-model behavior.

| Benchmark | Metric | Capitalized mixed | Matched cased | Matched uncased | Current read |
| --- | --- | ---: | ---: | ---: | --- |
| CoNLL-2003 NER | entity F1 | 0.915800 +/- 0.002013, n=5 | 0.913101 +/- 0.001531, n=5 | 0.902421 +/- 0.003327, n=5 | cap > cased > uncased |
| WNUT-17 NER | entity F1 | 0.452072 +/- 0.001681, n=5 | 0.441257 +/- 0.012031, n=5 | 0.437926 +/- 0.019077, n=5 | cap > cased ~= uncased |
| OntoNotes v5 NER | entity F1 | 0.887593 +/- 0.001624, n=5 | 0.888170 +/- 0.003293, n=5 | 0.872913 +/- 0.001348, n=5 | cap ~= cased > uncased |
| PTB POS | accuracy | 0.977179 +/- 0.000199, n=5 | 0.977100 +/- 0.000569, n=5 | 0.973149 +/- 0.000386, n=5 | cap ~= cased > uncased |

Interpretation:

- The primary "beats uncased where case matters" claim is supported on all four
  required token benchmarks.
- Bootstrap reports support significant cap-over-uncased wins on CoNLL,
  OntoNotes, and PTB. WNUT has a positive mean but remains high variance at
  five seeds.
- Against cased, the correct paper framing is "matches cased, sometimes wins";
  bootstrap intervals include zero for CoNLL, OntoNotes, and PTB, and WNUT is
  positive by mean but not significant at five seeds.

Bootstrap reports:

```text
/workspace/capitalization_embeddings/reports/mixed_case_token_ner_bootstrap_1000_fast.md
/workspace/capitalization_embeddings/reports/mixed_case_ptb_bootstrap_10000.md
```

## Other Completed Benchmarks

These are useful, but they should be positioned carefully because they contain
both positive and negative evidence.

| Benchmark | Metric | Capitalized mixed/current-best | Matched cased | Matched uncased | Current read |
| --- | --- | ---: | ---: | ---: | --- |
| SST-5 | accuracy | 0.544253 +/- 0.002548, n=5 | 0.528326 +/- 0.008475, n=5 | 0.539910 +/- 0.002922, n=5 | cap > uncased > cased, not significant |
| TweetEval Irony | macro-F1 | 0.675807 +/- 0.012560, n=5 | 0.656121 +/- 0.015872, n=5 | 0.667333 +/- 0.016625, n=5 | cap > uncased > cased, not significant |
| TweetEval Offensive | macro-F1 | 0.806396 +/- 0.006190, n=5 | 0.792561 +/- 0.004841, n=5 | 0.799580 +/- 0.010435, n=5 | cap > uncased > cased, not significant |
| 20 Newsgroups | accuracy | 0.704965 +/- 0.001271, n=5 | 0.693999 +/- 0.002180, n=5 | 0.706320 +/- 0.002480, n=5 | cap ~= uncased > cased |
| TREC Fine | accuracy | 0.820400 +/- 0.007925, n=5 | 0.836000 +/- 0.005099, n=5 | 0.829600 +/- 0.008877, n=5 | cap underperforms both |
| TweetEval Emoji | accuracy | 0.369396 +/- 0.002440, n=5 | 0.453884 +/- 0.002073, n=5 | 0.369372 +/- 0.003103, n=5 | cased dominates; cap ~= uncased |
| SciERC relations | accuracy | 0.844353 +/- 0.010585, n=5 | 0.843326 +/- 0.010832, n=5 | 0.861602 +/- 0.010174, n=5 | uncased dominates |
| Combined scientific relations | accuracy | 0.625282 +/- 0.004729, n=5 | 0.624680 +/- 0.006076, n=5 | 0.632506 +/- 0.010860, n=5 | uncased dominates |
| SemEval18 Task 7 validation | accuracy | 0.530081 +/- 0.040813, n=5 | 0.585366 +/- 0.043782, n=5 | 0.513821 +/- 0.049252, n=5 | cased dominates, high variance |
| SciEntsBank 3-way UQ | macro-F1 | 0.398351 +/- 0.011996, n=5 | 0.399589 +/- 0.027142, n=5 | 0.424502 +/- 0.027775, n=5 | uncased dominates |
| SciEntsBank 3-way UD | macro-F1 | 0.410593 +/- 0.008775, n=5 | 0.516709 +/- 0.021029, n=5 | 0.447222 +/- 0.043735, n=5 | cased dominates |

Interpretation:

- Positive sequence/classification evidence exists for SST-5, TweetEval Irony,
  TweetEval Offensive, and partial 20 Newsgroups.
- Scientific relation and SciEntsBank results do not currently support the
  broad "cap embeddings preserve uncased wins" claim.
- TREC Fine and TweetEval Emoji are useful negative controls. They should either
  be reported honestly as limitations or omitted from the main table and kept in
  an appendix if the paper's stated scope is token-level capitalization.

Bootstrap reports:

```text
/workspace/capitalization_embeddings/reports/mixed_case_sequence_macro_f1_bootstrap_10000.md
/workspace/capitalization_embeddings/reports/mixed_case_sequence_accuracy_bootstrap_10000.md
```

## Statistical Readiness

Current status:

- The current best mixed-case method now has 5 seeds on the required token
  benchmarks and four selected uncased-favored sequence benchmarks.
- Per-example prediction files are saved for completed runs, so paired bootstrap
  tests are feasible.
- Final paired bootstrap/Holm-corrected tables have not yet been generated.

Approximate seed-level/bootstrap status from current paired seed deltas:

| Comparison | Observed mean delta | Paired-seed SD | Rough n for 80% power | Read |
| --- | ---: | ---: | ---: | --- |
| mixed CoNLL cap - cased | +0.002699 | 0.002558 | 8 | bootstrap CI includes 0 |
| mixed CoNLL cap - uncased | +0.013379 | 0.003922 | 1 | bootstrap significant |
| mixed WNUT cap - cased | +0.010816 | 0.012729 | 11 | high variance; not significant |
| mixed WNUT cap - uncased | +0.014146 | 0.020252 | 17 | high variance; not significant |
| mixed OntoNotes cap - cased | -0.000577 | 0.003299 | 257 | tie/non-inferiority framing |
| mixed OntoNotes cap - uncased | +0.014680 | 0.002421 | 1 | bootstrap significant |
| mixed PTB cap - cased | +0.000079 | 0.000677 | 575 | tie/non-inferiority framing |
| mixed PTB cap - uncased | +0.004030 | 0.000397 | 1 | bootstrap significant |

These are rough seed-count estimates, not final significance tests. Final
claims should use both:

1. Seed-level replication: report mean, standard deviation, and paired seed
   deltas.
2. Example-level uncertainty: paired bootstrap over the saved predictions.

For top-tier claims, use these labels:

- `win`: mean delta positive and bootstrap CI excludes 0.
- `tie`: bootstrap CI includes 0 or absolute delta is below a predeclared
  practical threshold.
- `loss`: mean delta negative and bootstrap CI excludes 0.

Predeclare practical equivalence thresholds before the final sweep:

```text
F1/accuracy headline threshold: 0.002 absolute
Regression/correlation threshold: 0.002 absolute
```

## What Is Still Needed For An Iron-Clad Submission

Minimum next steps:

1. Report Holm-Bonferroni-corrected significance within the token-task family.
2. Add an ablation table comparing:
   - 3-class capitalization embeddings
   - 4-class mixed-case capitalization embeddings
   - 4-class + capitalization embedding dropout
   - matched cased and matched uncased controls
3. Add parameter/memory accounting:
   - cased vocabulary embedding parameters
   - uncased vocabulary embedding parameters
   - capitalization embedding overhead
   - actual saved model size and training/inference throughput if available
4. Add an error analysis:
   - first-cap entities
   - all-caps acronyms
   - mixed-case words such as `iPhone`
   - tokens where cased wins and cap embeddings lose
5. Write a locked experimental protocol:
   - datasets and splits
   - model-selection rule
   - seeds
   - pretraining recipe
   - downstream hyperparameters
   - significance tests
   - which benchmarks are primary vs appendix/diagnostic

Best compute allocation:

- Run up to 20-30 seeds only for the high-variance headline comparisons, mainly
  WNUT and TweetEval-style tasks.
- Use 5-10 seeds for low-variance tasks such as OntoNotes and PTB, where the
  result is already stable.
- Do not spend 20-30 seeds trying to turn stable losses into wins. OntoNotes
  and PTB currently support "near-cased / equivalent within a small margin,"
  not "beats cased."

## Recommended Paper Claim

Recommended main claim:

```text
Capitalization embeddings add a small learned case channel to an uncased BERT
vocabulary. In matched continued-pretraining controls, this recovers nearly all
of the cased-model advantage on capitalization-sensitive token tasks while
retaining the parameter sharing of the uncased vocabulary.
```

Recommended claims to avoid unless future runs change the evidence:

- "Capitalization embeddings strictly dominate cased BERT."
- "Capitalization embeddings strictly dominate uncased BERT on all noisy-text
  or scientific-text benchmarks."
- "The method universally improves BERT."

## Immediate Work Queue

1. Apply Holm-Bonferroni correction to the generated bootstrap p-values.
2. Decide whether WNUT should be expanded to 20-30 seeds because the observed
   cap-vs-cased delta is positive but high variance.
3. Add ablation runs for mixed-case/dropout vs 3-class under the same seeds.
4. Convert this ledger into a paper table and appendix table once final stats
   are available.
