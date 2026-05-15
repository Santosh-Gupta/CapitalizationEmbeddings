# Benchmark Sweep Summary

Metric: `macro_f1`

| Benchmark | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased |
| --- | ---: | ---: | ---: | ---: | ---: |
| isarcasm_eval_en | 0.6076 +/- 0.0175 | 0.6075 +/- 0.0109 | 0.6032 +/- 0.0174 | -0.0044 | -0.0043 |

## Comparisons

### isarcasm_eval_en

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | -0.004397 | [-0.043937, +0.045050] | 0.7316 |
| capitalized_pretrained | cased_pretrained | -0.004345 | [-0.050490, +0.049391] | 0.8348 |
| cased_pretrained | uncased_pretrained | -0.000052 | [-0.033227, +0.045177] | 0.8508 |
