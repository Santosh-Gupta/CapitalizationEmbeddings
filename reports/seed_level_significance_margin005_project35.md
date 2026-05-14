# Seed-Level Significance

Alpha: `0.05`
Non-inferiority/equivalence margin: `0.005`
Projected n: `35`

This report treats random seed as the replicated unit. Superiority is a
one-sided paired-seed test that the capitalized model is better. Matching is
reported as non-inferiority to the better baseline; strict equivalence uses
TOST and is harder because it also rejects being meaningfully better.

| Benchmark | Baseline | Best? | n | Delta | CI95 | Sup p | Non-inf p | Non-inf | Equiv p | Equiv | Projected CI95 | Projected sup | Projected non-inf | n80 sup | n80 non-inf |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | --- | ---: | ---: |
| conll2003_ner | uncased | no | 20 | +0.012525 | [+0.011200, +0.013851] | 1.943e-14 | 0 | yes | 1 | no | [+0.011553, +0.013498] | yes | yes | 1 | 1 |
| conll2003_ner | cased | yes | 20 | +0.004537 | [+0.002535, +0.006539] | 7.069e-05 | 2.758e-09 | yes | 0.3169 | no | [+0.003068, +0.006006] | yes | yes | 7 | 2 |
| wnut17_ner | uncased | no | 20 | +0.007103 | [-0.000372, +0.014578] | 0.03066 | 0.001541 | yes | 0.7185 | no | [+0.001616, +0.012589] | yes | yes | 40 | 11 |
| wnut17_ner | cased | yes | 20 | +0.006910 | [+0.000707, +0.013112] | 0.01544 | 0.0003668 | yes | 0.7365 | no | [+0.002357, +0.011462] | yes | yes | 29 | 8 |
| tweet_eval_irony | uncased | yes | 20 | +0.008566 | [+0.001487, +0.015645] | 0.01015 | 0.0003737 | yes | 0.8475 | no | [+0.003370, +0.013762] | yes | yes | 25 | 8 |
| tweet_eval_irony | cased | no | 20 | +0.019679 | [+0.011885, +0.027474] | 2.114e-05 | 1.217e-06 | yes | 0.9996 | no | [+0.013958, +0.025400] | yes | yes | 6 | 3 |
| tweet_eval_offensive | uncased | yes | 20 | +0.006566 | [+0.003427, +0.009706] | 0.000162 | 1.442e-07 | yes | 0.8453 | no | [+0.004262, +0.008871] | yes | yes | 9 | 3 |
| tweet_eval_offensive | cased | no | 20 | +0.013305 | [+0.009421, +0.017190] | 4.108e-07 | 3.287e-09 | yes | 0.9999 | no | [+0.010454, +0.016157] | yes | yes | 4 | 2 |
| sst5 | uncased | yes | 20 | -0.000339 | [-0.003457, +0.002778] | 0.5889 | 0.002763 | yes | 0.002763 | yes | [-0.002628, +0.001949] | no | yes | inf | 13 |
| sst5 | cased | no | 20 | +0.012353 | [+0.007377, +0.017329] | 2.571e-05 | 3.182e-07 | yes | 0.997 | no | [+0.008701, +0.016005] | yes | yes | 6 | 3 |
| twenty_newsgroups | uncased | yes | 20 | -0.000843 | [-0.002507, +0.000821] | 0.8488 | 2.398e-05 | yes | 2.398e-05 | yes | [-0.002065, +0.000379] | no | yes | inf | 5 |
| twenty_newsgroups | cased | no | 20 | +0.010761 | [+0.009543, +0.011979] | 6.584e-14 | 0 | yes | 1 | no | [+0.009867, +0.011655] | yes | yes | 1 | 1 |
