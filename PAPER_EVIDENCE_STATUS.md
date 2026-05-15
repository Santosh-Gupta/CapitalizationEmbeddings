# Paper Evidence Status

Last updated: 2026-05-15

This file is the durable project ledger for paper readiness. It summarizes what
is supported by completed runs, what is still preliminary, and what must be run
before making top-tier conference or journal claims.

## Current Answer

We now have the main GPU-heavy 20-seed evidence package for the current best
method. It is not an "iron-clad universal dominance" result, but it is a
defensible paper package if the claim is scoped carefully:

```text
Capitalization embeddings recover most of the benefit of cased BERT on
capitalization-sensitive token tasks while keeping the lexical sharing of an
uncased vocabulary.
```

The current evidence is strongest for cased-favored token tasks. With 20 seeds,
the mixed-case capitalization variant beats matched uncased on CoNLL-2003 with
Holm-corrected significance and has positive but high-variance mean deltas on
WNUT-17. Against matched cased controls it is best framed as a tie/near-cased
result, not a statistically significant win.

For selected uncased-favored sequence tasks, the current best method does not
hurt uncased performance in a statistically detectable way: SST-5 and 20
Newsgroups are practical ties against uncased, and TweetEval Irony/Offensive are
positive by mean but not significant. Some other sequence/scientific diagnostics
remain negative, so avoid a broad "strictly better than cased and uncased BERT"
claim.

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

## Added Cased-Favored Candidates

The benchmark registry now includes these additional cased-favored candidates:

| Benchmark key | User-proposed source | Status | Caveat |
| --- | --- | --- | --- |
| `tweet_eval_emoji` | TweetEval Emoji | already implemented and 5-seed diagnostic completed | Clean broad benchmark, but current cap-embed result matches uncased and loses to cased. |
| `trec_fine` | TREC Fine | already implemented and 5-seed diagnostic completed | Clean replicated benchmark, but current cap-embed result underperforms both baselines. |
| `kaggle_walia_ner` | Kaggle/Walia NER | 5-seed diagnostic completed | Public HF mirror has only one train split, so runner creates deterministic train/validation/test splits per seed. Published BERT comparison used BERT as embeddings into another architecture, so this is supporting evidence rather than a clean BERT fine-tune replication. Current result: cap beats uncased and ties cased. |
| `isarcasm_eval_en` | iSarcasmEval original English Task A | 5-seed diagnostic completed | Small social sarcasm benchmark; useful as supporting evidence, but not a broad headline benchmark by itself. Current result: cap ties both baselines, with a small negative mean. |
| `citation_sentiment_acl` | Public citation sentiment corpus | implemented, not yet run | This is the ACL citation sentiment corpus, not the tiny 97-example ACM test set. Do not transfer the reported ACM gap without verifying and acquiring that exact test set. |

Implementation notes:

- `kaggle_walia_ner` runs through `scripts/run_token_classification_benchmark.py`.
- `isarcasm_eval_en` and `citation_sentiment_acl` run through
  `scripts/run_sequence_classification_benchmark.py`.
- `kaggle_walia_ner` and `isarcasm_eval_en` have 5-seed GPU diagnostics under
  `/workspace/capitalization_embeddings/checkpoints/added_cased_favored_5seed`.
  `citation_sentiment_acl` still has only loader smoke checks.

Regenerate an evidence ledger from the current JSONL files with:

```bash
python scripts/collect_evidence_status.py \
  --checkpoint-root /workspace/capitalization_embeddings/checkpoints
```

Generate paired-bootstrap summaries with `scripts/summarize_benchmark_sweep.py`,
then apply Holm-Bonferroni correction to one or more summary JSON files with:

```bash
python scripts/apply_holm_correction.py \
  /workspace/capitalization_embeddings/reports/<summary>.json \
  --family token_headline \
  --output-md /workspace/capitalization_embeddings/reports/<holm_report>.md \
  --output-json /workspace/capitalization_embeddings/reports/<holm_report>.json
```

Do not report `/workspace/capitalization_embeddings/checkpoints/scientific_5seed/semeval2018_task7`
as a final benchmark result; that run produced all-zero values. Use
`semeval2018_validation_5seed/semeval2018_task7` for the corrected validation
run.

## Theory And Efficiency Evidence

CPU-only reports:

```text
reports/parameter_efficiency.md
reports/vocab_fragmentation.md
```

Parameter-accounting result:

| Design | Encoder+pooler params | Extra vs uncased encoder | MLM params | Extra vs uncased MLM |
| --- | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | 109,482,240 | +0 | 109,514,298 | +0 |
| `bert-base-cased` | 108,310,272 | -1,171,968 | 108,340,804 | -1,173,494 |
| capitalized BERT, 4 case states | 109,485,312 | +3,072 | 109,520,446 | +6,148 |
| hypothetical 3x uncased case-expanded vocab | 156,364,032 | +46,881,792 | 156,457,134 | +46,942,836 |
| hypothetical 4x uncased case-expanded vocab | 179,804,928 | +70,322,688 | 179,928,552 | +70,414,254 |

Vocabulary-fragmentation result:

| Quantity | Count |
| --- | ---: |
| `bert-base-cased` first-cap/all-caps/mixed-case WordPieces | 8,368 |
| first-cap/all-caps cased WordPieces with lowercase uncased counterpart | 8,172 |
| cased-vocab alphabetic families with multiple case forms | 3,819 |

Paper-framing note:

- The efficiency claim should be made against explicit case-expanded vocabulary
  designs, not against released `bert-base-cased`, because `bert-base-cased` has
  a smaller vocabulary than `bert-base-uncased`.
- The vocabulary-fragmentation analysis supports the factorization hypothesis:
  many cased WordPieces can be represented as a lowercase lexical identity plus
  a small learned case feature.

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
| Kaggle/Walia NER | entity F1 | 0.842183 +/- 0.007401, n=5 | 0.843687 +/- 0.006108, n=5 | 0.826976 +/- 0.006008, n=5 | cap > uncased, cap ~= cased |
| iSarcasmEval original EN | macro-F1 | 0.603190 +/- 0.017442, n=5 | 0.607535 +/- 0.010922, n=5 | 0.607586 +/- 0.017493, n=5 | cap ties both; small negative mean |
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
- Kaggle/Walia NER is useful supporting evidence for the token-level
  capitalization claim: cap is +0.015207 F1 over uncased with bootstrap CI95
  [+0.008672, +0.021349], and ties cased under a 0.005 practical margin.
- iSarcasmEval is neutral evidence: cap is within the wide bootstrap intervals
  of both baselines, with no positive mean advantage.

Bootstrap reports:

```text
/workspace/capitalization_embeddings/reports/mixed_case_sequence_macro_f1_bootstrap_10000.md
/workspace/capitalization_embeddings/reports/mixed_case_sequence_accuracy_bootstrap_10000.md
/workspace/capitalization_embeddings/reports/added_cased_favored_5seed_walia_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/added_cased_favored_5seed_isarcasm_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/added_cased_favored_5seed_holm_margin005.md
```

## Statistical Readiness

Current status:

- The current best mixed-case method has 20 seeds on CoNLL-2003, WNUT-17, and
  four selected uncased-favored sequence benchmarks.
- Matched cased/uncased controls have 20 seeds for CoNLL-2003, WNUT-17,
  TweetEval Irony, TweetEval Offensive, SST-5, and 20 Newsgroups.
- OntoNotes v5 and PTB POS remain 5-seed results for current-best and matched
  controls; they are low-variance tie/non-inferiority evidence, not headline
  20-seed significance evidence.
- Per-example prediction files are saved for completed runs, so paired bootstrap
  tests are feasible.
- Final paired bootstrap/Holm-corrected tables have been generated for the
  current 20-seed token/sequence headline set.

Final 20-seed reports:

```text
/workspace/capitalization_embeddings/reports/final_token_20seed_ner_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/final_token_20seed_holm.md
/workspace/capitalization_embeddings/reports/final_sequence_20seed_macro_f1_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/final_sequence_20seed_accuracy_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/final_sequence_20seed_holm.md
reports/seed_level_significance_margin002_project35.md
reports/seed_level_significance_margin005_project35.md
```

20-seed headline table:

| Benchmark | Metric | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased | Holm label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CoNLL-2003 NER | entity F1 | 0.9040 +/- 0.0025 | 0.9119 +/- 0.0035 | 0.9165 +/- 0.0018 | +0.0125 | +0.0045 | win vs uncased, tie vs cased |
| WNUT-17 NER | entity F1 | 0.4424 +/- 0.0152 | 0.4426 +/- 0.0100 | 0.4495 +/- 0.0103 | +0.0071 | +0.0069 | tie vs both |
| TweetEval Irony | macro-F1 | 0.6702 +/- 0.0135 | 0.6591 +/- 0.0144 | 0.6788 +/- 0.0092 | +0.0086 | +0.0197 | tie vs both |
| TweetEval Offensive | macro-F1 | 0.8034 +/- 0.0091 | 0.7966 +/- 0.0073 | 0.8099 +/- 0.0065 | +0.0066 | +0.0133 | tie vs both |
| SST-5 | accuracy | 0.5410 +/- 0.0039 | 0.5283 +/- 0.0084 | 0.5407 +/- 0.0068 | -0.0003 | +0.0124 | tie vs both |
| 20 Newsgroups | accuracy | 0.7055 +/- 0.0028 | 0.6939 +/- 0.0023 | 0.7047 +/- 0.0031 | -0.0008 | +0.0108 | tie vs uncased; positive vs cased but not Holm-significant |

Holm-corrected significant result:

```text
CoNLL-2003 cap > uncased:
delta +0.012525, bootstrap CI95 [+0.003364, +0.019591],
raw p = 0.0086, Holm p = 0.0344.
```

Important non-significant but useful results:

```text
WNUT-17:
cap is positive by mean vs both baselines, but high variance keeps it a tie.

Selected uncased-favored sequence tasks:
cap matches uncased within the predeclared 0.002 practical threshold on SST-5
and 20 Newsgroups; it is positive by mean on TweetEval Irony/Offensive, but not
significant after bootstrap/Holm.
```

Seed-level significance and matching:

- Treat random seed as the replicated unit for the main fine-tuning stability
  analysis.
- Use superiority tests for "beats" claims.
- Use non-inferiority to the better baseline for "matches the best baseline"
  claims. Strict equivalence is stronger and is not the right label when the
  capitalized model is meaningfully better than one baseline.
- With a 0.002 margin, current 20-seed seed-level tests support non-inferiority
  to the better baseline for CoNLL, WNUT, TweetEval Irony, and TweetEval
  Offensive. They do not yet support 0.002-margin non-inferiority for SST-5 or
  20 Newsgroups.
- With a 0.005 margin, current 20-seed seed-level tests support
  non-inferiority to the better baseline for all six 20-seed headline
  benchmarks.
- Projecting from current 20-seed seed variances to 35 seeds, superiority is
  plausible for CoNLL-vs-cased, WNUT-vs-cased, TweetEval Irony-vs-uncased, and
  TweetEval Offensive-vs-uncased. A 0.002-margin non-inferiority claim for SST-5
  and 20 Newsgroups is not likely to be secured by 35 seeds; the projection is
  closer to roughly 100 and 59 seeds respectively.

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

Practical equivalence thresholds used for the final 20-seed Holm reports:

```text
F1/accuracy headline threshold: 0.002 absolute
Regression/correlation threshold: 0.002 absolute
```

## What Is Still Needed For An Iron-Clad Submission

Completed paper-hardening items now include:

- locked experimental protocol in `EXPERIMENTAL_PROTOCOL.md`;
- GPU ablation plan in `ABLATION_MATRIX.md`;
- paper-ready result tables in `reports/paper_tables.md`;
- parameter accounting in `reports/parameter_efficiency.md`;
- vocabulary-fragmentation accounting in `reports/vocab_fragmentation.md`;
- manuscript outline and draft scaffold in `MANUSCRIPT_OUTLINE.md` and
  `paper/draft.md`;
- citation/provenance tracking in `paper/references.bib`,
  `CITATION_STATUS.md`, and `DATASET_PROVENANCE.md`;
- error-analysis tooling in `scripts/error_analysis_by_case.py`.

Remaining before a genuinely strong submission:

1. Run the locked ablation table comparing:
   - 3-class capitalization embeddings;
   - 4-class mixed-case capitalization embeddings;
   - 4-class + capitalization embedding dropout;
   - matched cased and matched uncased controls;
   - no-capitalization-embedding/no-auxiliary-loss controls if budget allows.
2. Run token-level error analysis from saved prediction JSONL files:
   - first-cap entities;
   - all-caps acronyms;
   - mixed-case words such as `iPhone`;
   - tokens where cased wins and cap embeddings lose.
3. Add actual throughput/saved-model-size measurements if the final paper makes
   an efficiency claim beyond parameter counts.
4. Convert `paper/draft.md` into the target venue format after ablation and
   error-analysis results are available.

Best compute allocation:

- No more GPU sweeps are needed for the current decision gate.
- If more compute is purchased, spend it on ablations or method improvements,
  not on repeating the completed 20-seed selected sequence controls.
- WNUT remains high variance at 20 seeds. Expanding to 30 seeds may tighten the
  estimate, but the current result already supports a tie/positive-mean framing;
  it is unlikely to create an iron-clad universal dominance claim by itself.
- OntoNotes and PTB currently support "near-cased / equivalent within a small
  margin," not "beats cased."

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

1. Run error analysis once the RunPod network volume or saved prediction roots
   are accessible.
2. Run the locked ablations on CoNLL and Walia when the GPU pod is restarted.
3. Decide whether to expand Walia, OntoNotes, and PTB beyond 5 seeds only after
   ablations show the method story is clean.
4. Decide whether to run 30-35 seed WNUT only if the paper needs a
   WNUT-specific superiority claim; current evidence already supports
   tie/positive-mean framing.
