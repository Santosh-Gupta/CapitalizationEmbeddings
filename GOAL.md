# Project Goal

Build a paper-grade proof that capitalization embeddings preserve the main
advantage of uncased BERT while recovering most of the advantage of cased BERT
when capitalization matters.

## Main Claim

The target claim is no longer simply "beat cased BERT everywhere." The stronger
and more defensible claim is:

```text
Capitalization embeddings keep lexical sharing through an uncased vocabulary,
while adding a small case channel that downstream models can use when case is
predictive and ignore when case is noisy.
```

This predicts two benchmark families:

1. Cased-favored tasks: capitalized should beat matched uncased and match or beat
   matched cased.
2. Uncased-favored tasks: capitalized should match or beat matched uncased and
   outperform matched cased, or at least avoid the cased penalty.

## Current Best Method

Use the real-acronym 3k/lr2e-5 continued-pretraining recipe as the main
checkpoint family:

```text
corpus: capitalization_real_acronym_mix
max_steps: 3000
learning_rate: 2e-5
capitalization_loss_weight: 0.25
```

Matched controls must receive the same extra MLM recipe:

```text
capitalized: /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/capitalized_from_task_mix_steps3000_lr2e5/final
uncased:     /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final
cased:       /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final
```

## Current Evidence

Single-seed discovery results are encouraging but not paper-grade.

Cased-favored tasks:

- CoNLL-2003 NER: capitalized beats matched cased and uncased.
- WNUT-17 NER: capitalized beats matched cased and uncased.
- OntoNotes v5 NER: capitalized beats matched uncased and nearly matches cased.
- PTB POS: capitalized beats matched uncased and nearly matches cased.

Uncased-favored tasks:

- Capitalized beats both matched controls on TweetEval Irony, SST-5,
  20 Newsgroups, TweetEval Sentiment, and TweetEval Offensive.
- TweetEval Emotion is currently a miss against matched uncased.
- STS-B needs validation-split rerun because GLUE test labels are hidden.
- Yahoo Answers is pending because full fine-tuning is much larger than the
  other sequence tasks.

## Paper-Grade Completion Criteria

For every headline benchmark:

1. Run at least 5 fine-tuning seeds for each model family.
2. Save per-example predictions for every model/seed.
3. Report mean and standard deviation across seeds.
4. Run paired bootstrap tests on identical examples.
5. Apply Holm-Bonferroni correction within benchmark families.

Target final headline benchmark set:

- Cased-favored: CoNLL-2003 NER, WNUT-17 NER, OntoNotes v5 NER, PTB POS.
- Uncased-favored: TweetEval Irony, SST-5, 20 Newsgroups, TweetEval Sentiment,
  TweetEval Offensive, TweetEval Emotion, STS-B validation, Yahoo Answers.

## Active Work Queue

1. Make benchmark runners fully multi-seed-safe.
2. Run a 5-seed confirmation sweep on the strongest and cheapest tasks first:
   WNUT-17, CoNLL-2003, TweetEval Irony, TweetEval Offensive, SST-5, and
   20 Newsgroups.
3. Run paired bootstrap significance tests for the 5-seed sweep.
4. Rerun STS-B on validation with the regression fix.
5. Decide whether Yahoo Answers should be full-dataset, sampled, or omitted from
   the first submission table.
6. If the 5-seed sweep holds, expand to 10 seeds for headline tasks.
