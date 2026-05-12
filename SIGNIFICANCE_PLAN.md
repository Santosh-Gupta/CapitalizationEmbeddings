# Statistical Significance Plan

The current benchmark table is discovery-grade: it uses one fine-tuning seed per
model/task. For a top-tier conference submission, claims should be supported by
both training-seed replication and paired test-set uncertainty.

## Primary Comparisons

Use the real-acronym 3k/lr2e-5 continued-pretraining checkpoints as the main
method unless a later checkpoint beats it under matched controls.

Primary pairwise comparisons:

- `capitalized_pretrained` vs `uncased_pretrained`
- `capitalized_pretrained` vs `cased_pretrained`
- `cased_pretrained` vs `uncased_pretrained`

All compared models must have received the same continued-MLM recipe for the
experiment family being reported.

## Required Evidence

For each downstream benchmark:

1. Run at least 5 fine-tuning seeds for every compared model.
2. Save per-example predictions for every model/seed.
3. Report mean and standard deviation across seeds.
4. Report paired bootstrap confidence intervals on the selected test/eval split.
5. Report paired p-values for the key model comparisons.

Recommended final setting: 10 seeds for headline tables if compute allows.

## Tests

Use paired bootstrap over examples for all tasks:

- Sequence classification: accuracy or macro-F1.
- Token classification: entity-level seqeval F1.
- Regression: Pearson/Spearman on the labeled validation split when GLUE test
  labels are hidden.

For classification accuracy tasks, McNemar can be included as a supplementary
check, but paired bootstrap is easier to use consistently across F1, seqeval,
and regression.

## Multiple Comparisons

Pre-register the primary hypotheses before the final sweep:

- Cased-favored tasks: capitalized should outperform uncased and match or beat
  cased.
- Uncased-favored tasks: capitalized should match or beat uncased and outperform
  cased, or at minimum avoid the cased penalty.

Use Holm-Bonferroni correction within each benchmark family:

- Case-sensitive token tasks: CoNLL, WNUT, OntoNotes, PTB.
- Uncased-favored sequence tasks: TweetEval variants, SST-5, 20 Newsgroups,
  Yahoo Answers, STS-B.

## Interpretation Rules

Use these labels in the paper:

- `win`: mean delta positive and 95% paired bootstrap CI excludes 0.
- `tie`: 95% paired bootstrap CI includes 0 and absolute mean delta is below a
  pre-declared practical threshold.
- `loss`: mean delta negative and 95% paired bootstrap CI excludes 0.

Suggested practical thresholds:

- F1/accuracy: 0.002 absolute.
- Pearson/Spearman: 0.002 absolute.

These thresholds should be fixed before the final sweep.
