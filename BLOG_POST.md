# Factoring Capitalization Out of BERT Tokens

I started this project with a simple question:

> What if an uncased BERT model could keep lexical sharing, but recover
> capitalization through a tiny extra embedding?

The original BERT release came in two familiar flavors: `bert-base-cased` and
`bert-base-uncased`. The cased model preserves forms like `Tom`, `tom`, and
`TOM` as distinct tokenization behavior. The uncased model maps these toward the
same lexical path, which improves sharing but drops capitalization.

The idea here is to factor those apart. Keep the ordinary uncased WordPiece ID,
but attach a parallel capitalization ID:

```text
token: tom      lexical id: tom      case id: none
token: Tom      lexical id: tom      case id: first-cap
token: TOM      lexical id: tom      case id: all-caps
token: iPhone   lexical id: iphone   case id: mixed-case
```

Then the BERT input representation becomes:

```text
x_i = word_embedding_i
    + position_embedding_i
    + segment_embedding_i
    + capitalization_embedding_i
```

This is not a claim that casing was previously unsolved. It is a small
engineering experiment around representation factorization: can case be modeled
as a reusable feature instead of being baked into many separate lexical
embeddings?

## Why This Is Plausible

The released `bert-base-cased` vocabulary contains many tokens that look like
case variants rather than genuinely different lexical concepts.

From the tokenizer analysis in this repo:

| Quantity | Count |
| --- | ---: |
| `bert-base-cased` vocabulary size | 28,996 |
| `bert-base-uncased` vocabulary size | 30,522 |
| cased vocab tokens with first-cap/all-caps/mixed-case form | 8,368 |
| first-cap/all-caps cased tokens with uncased lowercase counterpart | 8,172 |
| alphabetic case families with multiple case forms in cased vocab | 3,819 |

That is the core motivation. A lot of the cased vocabulary can be interpreted as
something like:

```text
lowercase lexical item + case feature
```

So I built that directly.

## What The Model Adds

The four-state capitalization model uses:

```text
0 = no capitalization feature / lowercase / punctuation / special token
1 = first-cap
2 = all-caps
3 = mixed-case
```

For BERT-base hidden size 768, the capitalization embedding table is tiny:

```text
4 * 768 = 3,072 parameters
```

During masked-language-model continued pretraining, I also used an auxiliary
capitalization prediction head. That head predicts the capitalization class at
masked positions, and the input case ID is zeroed at those positions so the
model cannot simply copy the answer.

The important efficiency comparison is not against released `bert-base-cased`.
That model actually has a smaller vocabulary than `bert-base-uncased`. The
useful comparison is against an explicit case-expanded uncased vocabulary.

| Design | Extra word-embedding parameters |
| --- | ---: |
| four-state capitalization embeddings | 3,072 |
| hypothetical lowercase/first-cap/all-caps vocabulary expansion | 46,881,792 |
| hypothetical lowercase/first-cap/all-caps/mixed-case expansion | 70,322,688 |

So the factorized version is not a way to make BERT-base smaller than cased
BERT. It is a way to add case information without exploding the lexical
embedding table.

## Tokenization Details

The tokenizer is still the Hugging Face uncased BERT tokenizer. The additional
work happens alongside tokenization:

1. Look at the original text before uncasing.
2. Use tokenizer offsets, or `word_ids` for pre-tokenized datasets, to align
   source spans to WordPieces.
3. Assign every WordPiece a capitalization ID.
4. Feed both `input_ids` and `capitalization_ids` to the model.

Mixed-case forms like `iPhone` are not given new lexical IDs. They travel
through the lowercase lexical path, with the mixed-case embedding carrying the
case signal.

## Training Setup

I used matched continued pretraining controls:

```text
bert-base-uncased + continued MLM
bert-base-cased   + continued MLM
bert-base-uncased + capitalization embeddings + continued MLM + case loss
```

The point of the matched controls is to avoid giving the capitalization model
extra data exposure and then accidentally attributing the gain to the
architecture.

The best current checkpoint is the four-state mixed-case model with:

```text
capitalization loss weight: 0.25
capitalization class weights: [1, 2, 8, 4]
capitalization embedding dropout: 0.1
continued pretraining: 3,000 steps on the real-acronym mix
```

There is also a V3 pretraining plan in the repo for a more serious follow-up,
but for this writeup I am focusing on what was already run.

## Results

The short version:

- On token/entity tasks, capitalization embeddings often recover most of the
  cased-model advantage while beating uncased.
- On tasks where uncased BERT is usually better, the added case channel usually
  does not hurt much.
- On some sequence tasks, especially TweetEval Emoji and TREC Fine, this method
  does not solve the cased/uncased tradeoff.

Main result table:

| Family | Benchmark | Metric | Uncased | Cased | Capitalized | Cap-Uncased | Cap-Cased | Label |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| token/entity | CoNLL-2003 NER | entity F1 | 0.9040 +/- 0.0025, n=20 | 0.9119 +/- 0.0035, n=20 | 0.9165 +/- 0.0018, n=20 | +0.0125 | +0.0045 | win vs uncased; tie vs cased |
| token/entity | Kaggle/Walia NER | entity F1 | 0.8270 +/- 0.0060, n=5 | 0.8437 +/- 0.0061, n=5 | 0.8422 +/- 0.0074, n=5 | +0.0152 | -0.0015 | win vs uncased; tie vs cased |
| token/entity | WNUT-17 NER | entity F1 | 0.4424 +/- 0.0152, n=20 | 0.4426 +/- 0.0100, n=20 | 0.4495 +/- 0.0103, n=20 | +0.0071 | +0.0069 | tie |
| control | SST-5 | accuracy | 0.5410 +/- 0.0039, n=20 | 0.5283 +/- 0.0084, n=20 | 0.5407 +/- 0.0068, n=20 | -0.0003 | +0.0124 | tie |
| control | TweetEval Irony | macro-F1 | 0.6702 +/- 0.0135, n=20 | 0.6591 +/- 0.0144, n=20 | 0.6788 +/- 0.0092, n=20 | +0.0086 | +0.0197 | tie |
| control | TweetEval Offensive | macro-F1 | 0.8034 +/- 0.0091, n=20 | 0.7966 +/- 0.0073, n=20 | 0.8099 +/- 0.0065, n=20 | +0.0066 | +0.0133 | tie |
| control | 20 Newsgroups | accuracy | 0.7055 +/- 0.0028, n=20 | 0.6939 +/- 0.0023, n=20 | 0.7047 +/- 0.0031, n=20 | -0.0008 | +0.0108 | tie vs uncased |
| neutral | iSarcasmEval EN | macro-F1 | 0.6076 +/- 0.0175, n=5 | 0.6075 +/- 0.0109, n=5 | 0.6032 +/- 0.0174, n=5 | -0.0044 | -0.0043 | tie |

The strongest result is CoNLL-2003 NER: the capitalization model beats matched
uncased and is at least competitive with matched cased. Kaggle/Walia NER shows a
similar pattern, though with fewer seeds.

The important caveat is that this is not a universal best-of-both-worlds model.
TweetEval Emoji favored cased BERT strongly in our diagnostics, and the
capitalization model stayed much closer to uncased. TREC Fine was also a miss.

## What I Learned

### 1. The factorization works best when case is a reusable feature

NER and POS-style tasks are a natural fit. Proper nouns, organizations,
locations, and acronyms all benefit from a reusable capitalization signal.

### 2. It does not magically recover every advantage of cased tokenization

Some cased-model gains are not just "this token is capitalized." They may come
from the whole cased pretraining distribution, different vocabulary allocation,
or task-specific conventions. TweetEval Emoji is the cleanest example in this
project.

### 3. Continued pretraining fairness matters

It is very easy to accidentally compare:

```text
new architecture + more pretraining
```

against:

```text
old baseline + less pretraining
```

That comparison is not meaningful. The repo therefore uses matched continued
pretraining controls for the headline comparisons.

### 4. Capitalization accuracy is diagnostic, but not sufficient

The capitalization channel can improve on its own held-out objective without
automatically improving downstream tasks. The downstream benchmark is the real
test.

## Possible Follow-Up: Move Case Into Attention

The implementation in this project adds capitalization embeddings to the input
embedding stack. That is simple and Hugging Face-friendly, but it may not be the
only way to represent case. Modern decoder LLMs usually inject position through
attention, for example with rotary position embeddings, rather than by adding a
learned position vector to the token embedding. A similar idea could be tried
for capitalization.

The lowest-risk variant would be a learned attention-logit bias:

```text
attention_score(i, j) =
  q_i k_j / sqrt(d)
  + case_bias[head, case_i, case_j]
```

This would let each head learn patterns such as "all-caps tokens attend more to
other all-caps tokens" or "first-cap tokens attend to surrounding lowercase
context" without modifying the token representation itself.

A more RoPE-like variant would rotate query/key vectors by capitalization state:

```text
q_i' = R_case(case_i) q_i
k_j' = R_case(case_j) k_j
```

Then attention depends on both lexical content and relative case state. Because
rotations preserve vector norm, this might be a cleaner way to add case than
ordinary additive embeddings. This project did not test that idea, but it is the
most interesting next architecture experiment.

## Limitations

This was an engineering exploration, not a final benchmark paper.

Known limitations:

- Not all ablations were completed.
- Some datasets only have 5 seeds.
- The method underperforms or fails to improve on several sequence tasks.
- The current best checkpoint came from a short continued-pretraining recipe.
- A stronger V3 pretraining plan exists, but I am not treating it as part of the
  completed results.

The right claim is narrow:

> Capitalization embeddings are a lightweight way to add case information to an
> uncased BERT-style model, and they are especially useful for token/entity
> tasks where case behaves like a reusable feature.

The wrong claim would be:

> This replaces cased tokenization everywhere.

The experiments do not support that.

## Code

The repo contains:

- Hugging Face-compatible tokenizer/collator/model code.
- Matched continued-pretraining scripts.
- Token and sequence benchmark runners.
- Bootstrap/Holm summary scripts.
- Parameter-efficiency and vocabulary-fragmentation analyses.
- A V3 pretraining plan for anyone who wants to push the idea further.

Relevant files:

```text
capitalization_embeddings/tokenization.py
capitalization_embeddings/collator.py
capitalization_embeddings/modeling.py
capitalization_embeddings/trainer.py
scripts/run_mlm_pretraining.py
reports/paper_tables.md
reports/parameter_efficiency.md
reports/vocab_fragmentation.md
PRETRAINING_V3_PLAN.md
```

## Bottom Line

This project did not produce a universal replacement for cased BERT. It did
produce evidence for a useful and intuitive representation trick:

```text
shared lowercase lexical embedding + tiny learned case embedding
```

For token/entity-heavy tasks, that trick can recover much of the benefit of
casing while preserving uncased lexical sharing. For broader sequence tasks, the
story is mixed, which is itself useful: capitalization is sometimes a reusable
feature, and sometimes part of a larger distributional package that a small
case embedding does not capture.
