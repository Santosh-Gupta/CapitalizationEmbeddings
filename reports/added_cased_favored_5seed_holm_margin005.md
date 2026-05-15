# Holm-Corrected Benchmark Significance

Family: `added_cased_favored_5seed`
Alpha: `0.05`
Practical equivalence threshold: `0.005`

| Benchmark | Comparison | n | Delta | CI95 | raw p | Holm p | Holm reject | Label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| isarcasm_eval_en | capitalized_pretrained>cased_pretrained | 5 | -0.004345 | [-0.050490, +0.049391] | 0.8348 | 1 | no | tie |
| isarcasm_eval_en | capitalized_pretrained>uncased_pretrained | 5 | -0.004397 | [-0.043937, +0.045050] | 0.7316 | 1 | no | tie |
| kaggle_walia_ner | capitalized_pretrained>cased_pretrained | 5 | -0.001504 | [-0.009226, +0.004954] | 0.7648 | 1 | no | tie |
| kaggle_walia_ner | capitalized_pretrained>uncased_pretrained | 5 | +0.015207 | [+0.008672, +0.021349] | 0 | 0 | yes | win |
