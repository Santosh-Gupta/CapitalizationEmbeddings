# Benchmark Sweep Summary

Metric: `seqeval_f1`

| Benchmark | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased |
| --- | ---: | ---: | ---: | ---: | ---: |
| kaggle_walia_ner | 0.8270 +/- 0.0060 | 0.8437 +/- 0.0061 | 0.8422 +/- 0.0074 | +0.0152 | -0.0015 |

## Comparisons

### kaggle_walia_ner

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | +0.015207 | [+0.008672, +0.021349] | 0 |
| capitalized_pretrained | cased_pretrained | -0.001504 | [-0.009226, +0.004954] | 0.7648 |
| cased_pretrained | uncased_pretrained | +0.016711 | [+0.011193, +0.022043] | 0 |
