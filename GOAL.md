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

Use the mixed-case/dropout continuation as the current best capitalization
embedding checkpoint family. It starts from the real-acronym 3k/lr2e-5
checkpoint and grows the capitalization feature vocabulary from three states to
four states:

```text
corpus: capitalization_real_acronym_mix
max_steps: 3000
learning_rate: 2e-5
capitalization_loss_weight: 0.25
capitalization_class_weights: [1, 2, 8, 4]
capitalization_embedding_dropout: 0.1
capitalization states: none, first-cap, all-caps, mixed-case
```

Matched controls must receive the same extra MLM recipe:

```text
capitalized: /workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final
uncased:     /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final
cased:       /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final
```

When expanding the capitalized model from three to four capitalization states,
rows 0-2 of the capitalization embedding/classifier tables are restored from
the 3-class checkpoint; only the new mixed-case row is newly initialized.

## Current Evidence

Single-seed and 3/5-seed discovery results are encouraging but not paper-grade
until paired-bootstrap/Holm-corrected summaries are generated.

Cased-favored tasks:

- CoNLL-2003 NER: mixed-case capitalized beats matched cased and uncased.
- WNUT-17 NER: mixed-case capitalized beats matched cased and uncased.
- OntoNotes v5 NER: mixed-case capitalized beats matched uncased and nearly
  matches cased.
- PTB POS: mixed-case capitalized beats matched uncased and nearly matches
  cased.

Uncased-favored tasks:

- Capitalized beats both matched controls on TweetEval Irony, SST-5,
  20 Newsgroups, TweetEval Sentiment, and TweetEval Offensive.
- TweetEval Emotion is currently a miss against matched uncased.
- Scientific relation classification is now a high-priority uncased-favored
  family. Reported gaps favor uncased BERT by about 2.8-3.0 points on SemEval18
  scientific relation classification and combined SemEval18+SciERC relation
  classification, making these better proof targets than many general sentiment
  tasks.
- SciEntsBank automatic short-answer grading is a high-value related benchmark
  family, especially 3-way unseen-question/unseen-domain variants where reported
  uncased gains are very large. Treat the SciEntsBank variants as related slices,
  not independent datasets, when writing the paper.
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
  Optional extra cased-favored sequence tasks: TREC Fine and TweetEval Emoji.
- Uncased-favored: TweetEval Irony, SST-5, 20 Newsgroups, TweetEval Sentiment,
  TweetEval Offensive, SemEval18 scientific relation classification,
  combined SemEval18+SciERC scientific relation classification, SciEntsBank
  3-way TUQ/TUD and 5-way TUQ/TUD, TweetEval Emotion, STS-B validation,
  Yahoo Answers.

Additional candidates:

- HASOC 2021 English Hate-Offensive Task A is a strong conceptual fit for the
  "uncased wins on noisy social text" claim, but official data access appears
  gated. Add it if the dataset files or a reliable public mirror are available.
- IWSLT 2012 TED punctuation restoration is useful only as a diagnostic
  lexical-sharing benchmark because the corpus is explicitly lowercased in the
  reported setup. It should not be used as primary evidence for capitalization
  embeddings unless the experiment is reframed.

## Active Work Queue

1. Expand the mixed-case token-task sweep from 3 seeds to at least 5 seeds for
   CoNLL-2003, WNUT-17, OntoNotes v5, and PTB POS.
2. Run paired bootstrap significance tests for the completed 3/5-seed sweeps.
3. Add ablations for 3-class versus 4-class mixed-case versus mixed-case +
   capitalization embedding dropout.
4. Decide which sequence/scientific benchmarks belong in the main paper table
   versus appendix/negative-control tables.
5. If the headline token-task sweep holds, expand WNUT and any high-variance
   headline tasks to 20-30 seeds.
