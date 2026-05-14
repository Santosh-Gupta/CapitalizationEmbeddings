# Benchmark Sweep Summary

Metric: `accuracy`

| Benchmark | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased |
| --- | ---: | ---: | ---: | ---: | ---: |
| sst5 | 0.5410 +/- 0.0039 | 0.5283 +/- 0.0084 | 0.5407 +/- 0.0068 | -0.0003 | +0.0124 |
| twenty_newsgroups | 0.7055 +/- 0.0028 | 0.6939 +/- 0.0023 | 0.7047 +/- 0.0031 | -0.0008 | +0.0108 |

## Comparisons

### sst5

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | -0.000339 | [-0.021267, +0.018100] | 1 |
| capitalized_pretrained | cased_pretrained | +0.012353 | [-0.015837, +0.039819] | 0.3839 |
| cased_pretrained | uncased_pretrained | -0.012692 | [-0.037557, +0.010407] | 0.3279 |

### twenty_newsgroups

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | -0.000843 | [-0.009825, +0.008364] | 0.8316 |
| capitalized_pretrained | cased_pretrained | +0.010761 | [+0.001328, +0.019384] | 0.0285 |
| cased_pretrained | uncased_pretrained | -0.011604 | [-0.021110, -0.001460] | 0.0252 |
