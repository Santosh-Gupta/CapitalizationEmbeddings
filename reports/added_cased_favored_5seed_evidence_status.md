# Evidence Status

Checkpoint root: `/workspace/capitalization_embeddings/checkpoints`

## added_cased_favored_5seed

added cased-favored benchmark diagnostics.

| Task | Metric | Model | n | Mean | Std | Seeds |
| --- | --- | --- | ---: | ---: | ---: | --- |
| isarcasm_eval_en | macro_f1 | capitalized_pretrained | 5 | 0.603190 | 0.017442 | 13,21,34,55,89 |
| isarcasm_eval_en | macro_f1 | cased_pretrained | 5 | 0.607535 | 0.010922 | 13,21,34,55,89 |
| isarcasm_eval_en | macro_f1 | uncased_pretrained | 5 | 0.607586 | 0.017493 | 13,21,34,55,89 |
| kaggle_walia_ner | seqeval_f1 | capitalized_pretrained | 5 | 0.842183 | 0.007401 | 13,21,34,55,89 |
| kaggle_walia_ner | seqeval_f1 | cased_pretrained | 5 | 0.843687 | 0.006108 | 13,21,34,55,89 |
| kaggle_walia_ner | seqeval_f1 | uncased_pretrained | 5 | 0.826976 | 0.006008 | 13,21,34,55,89 |

