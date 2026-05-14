# Benchmark Sweep Summary

Metric: `macro_f1`

| Benchmark | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased |
| --- | ---: | ---: | ---: | ---: | ---: |
| tweet_eval_irony | 0.6702 +/- 0.0135 | 0.6591 +/- 0.0144 | 0.6788 +/- 0.0092 | +0.0086 | +0.0197 |
| tweet_eval_offensive | 0.8034 +/- 0.0091 | 0.7966 +/- 0.0073 | 0.8099 +/- 0.0065 | +0.0066 | +0.0133 |

## Comparisons

### tweet_eval_irony

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | +0.008566 | [-0.028787, +0.047221] | 0.6442 |
| capitalized_pretrained | cased_pretrained | +0.019679 | [-0.020084, +0.062353] | 0.3834 |
| cased_pretrained | uncased_pretrained | -0.011114 | [-0.053105, +0.033207] | 0.6296 |

### tweet_eval_offensive

| A | B | Seed Delta | Bootstrap CI95 | p |
| --- | --- | ---: | ---: | ---: |
| capitalized_pretrained | uncased_pretrained | +0.006566 | [-0.018079, +0.029941] | 0.5775 |
| capitalized_pretrained | cased_pretrained | +0.013305 | [-0.015279, +0.043573] | 0.3844 |
| cased_pretrained | uncased_pretrained | -0.006739 | [-0.038945, +0.025538] | 0.6855 |
