# Dataset Provenance

This file records how each benchmark enters the paper. It should be updated
whenever a benchmark is promoted, demoted, or rerun.

## Main Token/Entity Benchmarks

| Benchmark key | Paper name | Loader | Split status | Citation key | Paper role |
| --- | --- | --- | --- | --- | --- |
| `conll2003_ner` | CoNLL-2003 NER | `lhoestq/conll2003` | Hugging Face mirror of canonical train/dev/test | `tjong-kim-sang-de-meulder-2003-introduction` | Main cased-favored token/entity evidence |
| `wnut17_ner` | WNUT-17 NER | `flaitenberger/wnut_17` | Hugging Face mirror of canonical train/dev/test | `derczynski-etal-2017-results` | Main noisy token/entity evidence |
| `ontonotes5_ner` | OntoNotes v5 NER | `extraordinarylab/ontonotes5` | Hugging Face mirror; verify label/split mapping before final submission | `weischedel-etal-2013-ontonotes` | Main token/entity replication |
| `ptb_pos` | Penn Treebank POS | `batterydata/pos_tagging` | Hugging Face mirror; verify PTB/WSJ provenance before final submission | `marcus-etal-1993-penn-treebank` | Main POS capitalization replication |
| `kaggle_walia_ner` | Kaggle/Walia NER | `rjac/kaggle-entity-annotated-corpus-ner-dataset` | Single train split; project creates deterministic train/validation/test splits per seed | `rjac-2022-kaggle-entity-annotated-corpus`; optionally `bos-etal-2017-groningen` | Supporting token/entity evidence, not primary canonical evidence |

## Uncased-Favored Controls

| Benchmark key | Paper name | Loader | Split status | Citation key | Paper role |
| --- | --- | --- | --- | --- | --- |
| `sst5` | SST-5 | `SetFit/sst5` | Hugging Face mirror of SST fine-grained splits | `socher-etal-2013-recursive` | Control where case is weakly predictive |
| `twenty_newsgroups` | 20 Newsgroups | `SetFit/20_newsgroups` | Hugging Face mirror; standard train/test style topic split | `lang-1995-newsweeder` | Control for lexical sharing/topic classification |
| `tweet_eval_irony` | TweetEval Irony | `tweet_eval/irony` | Hugging Face canonical TweetEval split | `barbieri-etal-2020-tweeteval` | Noisy social-text uncased-favored control |
| `tweet_eval_offensive` | TweetEval Offensive | `tweet_eval/offensive` | Hugging Face canonical TweetEval split | `barbieri-etal-2020-tweeteval` | Noisy social-text uncased-favored control |

## Negative Controls And Appendix Diagnostics

| Benchmark key | Paper name | Loader | Split status | Citation key | Paper role |
| --- | --- | --- | --- | --- | --- |
| `tweet_eval_emoji` | TweetEval Emoji | `tweet_eval/emoji` | Hugging Face canonical TweetEval split | `barbieri-etal-2020-tweeteval` | Negative control where cased dominates |
| `trec_fine` | TREC Fine | `lukasgarbas/trec` | Hugging Face mirror; verify fine-label mapping before final submission | `li-roth-2002-learning` | Negative-control cased-favored sequence task |
| `isarcasm_eval_en` | iSarcasmEval English Task A | `iabufarha/iSarcasmEval` | Hugging Face mirror of original task; small benchmark | `abu-farha-etal-2022-semeval` | Supporting social-text diagnostic |
| `semeval2018_task7` | SemEval-2018 Task 7 scientific relations | `DFKI-SLT/SemEval2018_Task7`, `Subtask_1_1` | Hugging Face mirror; project flattens relation rows | `gabor-etal-2018-semeval` | Appendix diagnostic |
| `scierc_relations` | SciERC relations | `nsusemiehl/SciERC` | Hugging Face mirror; verify relation preprocessing before final submission | `luan-etal-2018-multi` | Appendix diagnostic |
| `scientific_relations_combined` | Combined SemEval18 + SciERC relations | Project-combined loader | Project-created combined benchmark | `gabor-etal-2018-semeval`; `luan-etal-2018-multi` | Appendix diagnostic only |
| `scientbank_3way_uq` | SciEntsBank 3-way unseen-question | `nkazi/SciEntsBank` | Hugging Face mirror; project normalizes labels to 3-way | `dzikovska-etal-2013-semeval` | Appendix diagnostic |
| `scientbank_3way_ud` | SciEntsBank 3-way unseen-domain | `nkazi/SciEntsBank` | Hugging Face mirror; project normalizes labels to 3-way | `dzikovska-etal-2013-semeval` | Appendix diagnostic |
| `scientbank_5way_uq` | SciEntsBank 5-way unseen-question | `nkazi/SciEntsBank` | Hugging Face mirror; project uses 5-way labels | `dzikovska-etal-2013-semeval` | Appendix diagnostic |
| `scientbank_5way_ud` | SciEntsBank 5-way unseen-domain | `nkazi/SciEntsBank` | Hugging Face mirror; project uses 5-way labels | `dzikovska-etal-2013-semeval` | Appendix diagnostic |
| `citation_sentiment_acl` | ACL citation sentiment | `gaof23/citation_sentiment_corpus` | Implemented but not run; do not report until evaluated | `athar-2011-sentiment` | Candidate diagnostic only |

## Final-Submission Checks

Before submission:

1. Verify every Hugging Face mirror still resolves to the intended upstream
   dataset and license.
2. For `ontonotes5_ner`, `ptb_pos`, `trec_fine`, and `scierc_relations`, verify
   that label names and splits match the cited benchmark.
3. In the paper, mark project-created splits explicitly for
   `kaggle_walia_ner` and `scientific_relations_combined`.
4. Do not cite reported cased/uncased gaps from other papers as if they were our
   own results. Use them only to motivate benchmark selection.
