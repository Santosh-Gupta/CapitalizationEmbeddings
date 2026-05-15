# Factorized Capitalization Embeddings for Lexically Shared BERT Models

## Abstract

Cased language models preserve capitalization but can fragment lexical evidence
across surface-case variants, while uncased language models share lexical
evidence but discard case information. We study a simple factorization for
BERT-style encoders: keep the uncased WordPiece vocabulary, and add a small
learned capitalization embedding to each token. The resulting representation
separates lexical identity from case state, allowing forms such as `tom`,
`Tom`, and `TOM` to share a word embedding while retaining a learned case
feature. We continue-pretrain matched uncased, cased, and capitalization-aware
BERT models and evaluate them across capitalization-sensitive token/entity
tasks, uncased-favored controls, and negative-control benchmarks. On token and
entity tasks, capitalization embeddings recover much of the cased-model
advantage while substantially outperforming matched uncased controls on several
benchmarks. The method adds only 3,072 downstream parameters for four case
states, compared with 46.9M additional word-embedding parameters for a
hypothetical three-way case-expanded uncased vocabulary. The gains are not
universal: sequence tasks such as TweetEval Emoji and TREC Fine expose
limitations. These results support capitalization embeddings as a targeted,
parameter-efficient mechanism for case-sensitive token/entity modeling rather
than a universal replacement for cased tokenization.

## 1. Introduction

Tokenization in BERT-style encoders exposes a familiar tension. Cased tokenizers
preserve surface capitalization, which is often informative for named entities,
acronyms, and proper nouns. Uncased tokenizers collapse those surface variants,
which improves lexical sharing but removes a potentially useful signal. In the
standard design, this choice is made at the vocabulary level. A cased vocabulary
can allocate different embeddings to `Apple` and `apple`; an uncased vocabulary
uses one lexical embedding and loses the distinction.

This paper asks whether that tradeoff is necessary. Many cased WordPieces are
not distinct lexical concepts; they are predictable case variants of an
underlying lexical item. If the lexical item and the case pattern are separated,
the model can share lexical evidence while still using capitalization when it is
predictive.

We introduce factorized capitalization embeddings. The tokenizer emits the
ordinary uncased WordPiece ID plus a parallel capitalization ID. The model adds a
learned capitalization embedding to the usual BERT embedding sum. This adds a
tiny case channel without expanding the lexical vocabulary.

Our contributions are:

1. We implement a Hugging Face-compatible BERT variant that accepts
   capitalization IDs for continued pretraining and downstream fine-tuning.
2. We quantify case fragmentation in the released BERT cased vocabulary and show
   that thousands of cased WordPieces can be represented as lowercase
   WordPieces plus a small case feature.
3. We compare matched uncased, cased, and capitalization-aware models after the
   same continued-pretraining recipe.
4. We evaluate across token/entity tasks, uncased-favored controls, and
   negative-control tasks, using seed-level replication and paired bootstrap
   summaries.
5. We report limitations explicitly, showing that the method is strongest on
   token/entity capitalization tasks and not uniformly better on all sequence
   benchmarks.

## 2. Motivation: Case Fragmentation

The factorization hypothesis depends on a concrete vocabulary observation: many
tokens in a cased vocabulary are surface-case variants of forms that already
exist in an uncased vocabulary.

Using the released `bert-base-cased` and `bert-base-uncased` tokenizers, we find
that `bert-base-cased` contains 8,368 first-cap, all-caps, or mixed-case
WordPieces. Among first-cap and all-caps cased WordPieces, 8,172 have lowercase
counterparts in the uncased vocabulary. The cased vocabulary also contains 3,819
alphabetic case families with multiple case forms.

These counts support the view that cased tokenization often stores case
information in the lexical embedding table. Capitalization embeddings move that
information to a separate, low-dimensional categorical channel.

## 3. Method

For token position `i`, standard BERT builds an input representation from word,
position, and segment embeddings. We add a learned capitalization embedding:

```text
x_i = E_word(w_i) + E_position(i) + E_segment(s_i) + E_case(c_i)
```

Here `w_i` is the uncased WordPiece ID and `c_i` is a small capitalization
feature. The current model uses four case states:

```text
0 = no capitalization feature / lowercase / punctuation / special token
1 = first-cap
2 = all-caps
3 = mixed-case
```

The first-cap state covers examples such as `Tom`, the all-caps state covers
examples such as `NASA`, and the mixed-case state covers examples such as
`iPhone`, `eBay`, and `McDonald`. Mixed-case forms were collapsed to the
lowercase lexical path in the original three-state idea; the four-state variant
keeps a case signal without allocating new lexical embeddings.

Capitalization IDs are computed before uncased normalization. For raw text, we
use fast-tokenizer offsets to recover source spans. For pre-tokenized
sequence-labeling datasets, we use tokenizer `word_ids` and assign each
WordPiece from a source word the same capitalization ID.

### Continued Pretraining

We initialize from `bert-base-uncased`. All compatible BERT parameters are
loaded from the base checkpoint, while capitalization embeddings are newly
initialized. During masked language-model continued pretraining, we optimize:

```text
L = L_MLM + lambda L_case
```

The auxiliary capitalization head predicts the case class at masked positions.
At those masked positions, the input capitalization ID is zeroed so the model
cannot simply copy the case feature.

The current best checkpoint uses `lambda = 0.25`, capitalization class weights
`[1, 2, 8, 4]`, capitalization embedding dropout `0.1`, and 3,000 continued
pretraining steps at learning rate `2e-5` on the real-acronym mix.

### Matched Baselines

To isolate architecture from additional data exposure, all main comparisons use
matched continued-pretraining controls. The uncased baseline is
`bert-base-uncased` continued on the same corpus. The cased baseline is
`bert-base-cased` continued on the same corpus with the same recipe. The
capitalized model is compared against those matched controls.

## 4. Parameter Efficiency

The proposed model adds one capitalization embedding table. With BERT-base
hidden size 768 and four case states, that table contains:

```text
4 * 768 = 3,072 parameters
```

The auxiliary capitalization prediction head used during MLM pretraining adds
3,076 parameters. By contrast, explicitly tripling the uncased vocabulary to
allocate separate lowercase, first-cap, and all-caps embeddings would add
46,881,792 word-embedding parameters. A four-way expansion that also allocates a
mixed-case lexical variant would add 70,322,688 word-embedding parameters.

We do not claim that the proposed model is smaller than the released
`bert-base-cased` checkpoint. That checkpoint has a smaller vocabulary than
`bert-base-uncased`. The efficiency claim is against explicit case-expanded
vocabulary designs.

## 5. Experimental Setup

We evaluate three model families:

| Model key | Description |
| --- | --- |
| `uncased_pretrained` | matched `bert-base-uncased` continuation |
| `cased_pretrained` | matched `bert-base-cased` continuation |
| `capitalized_pretrained` | `bert-base-uncased` plus capitalization embeddings |

Token-classification tasks use 3 epochs, batch size 16, and learning rate
`3e-5`. Sequence-classification tasks use 3 epochs, batch size 16, and learning
rate `2e-5`. We save per-example predictions for paired bootstrap analysis.

We group benchmarks into three families:

1. token/entity capitalization tasks;
2. uncased-favored controls;
3. negative controls and appendix diagnostics.

We use superiority tests for "beats" claims and non-inferiority language for
"matches" claims. We report seed means, seed standard deviations, paired
bootstrap intervals, and Holm-corrected labels within benchmark families.

## 6. Results

The generated result table is in `reports/paper_tables.md`. The main pattern is
strongest on token/entity tasks.

On CoNLL-2003 NER, capitalization embeddings reach 0.9165 entity F1 across 20
seeds, compared with 0.9040 for matched uncased and 0.9119 for matched cased.
The cap-over-uncased delta is +0.0125 and is Holm-significant; the cap-over-cased
delta is positive but labeled as a tie.

On Kaggle/Walia NER, capitalization embeddings reach 0.8422 entity F1 across 5
seeds, compared with 0.8270 for matched uncased and 0.8437 for matched cased.
The cap-over-uncased delta is +0.0152 with bootstrap CI95
`[+0.0087, +0.0213]`, while the cap-vs-cased result is a practical tie.

WNUT-17 is high variance. The capitalization model has a positive mean over both
matched baselines, but intervals remain wide and the result is best described as
a tie with positive mean.

On uncased-favored controls, capitalization embeddings generally preserve
uncased behavior. On SST-5 and 20 Newsgroups, the model ties matched uncased.
On TweetEval Irony and Offensive, it has positive mean deltas over both
baselines, but not Holm-significant wins.

## 7. Ablations

The strongest remaining experimental gap is ablation. The planned ablation
matrix is defined in `ABLATION_MATRIX.md`. The most important ablations are:

- four-state capitalization embeddings with versus without capitalization
  embedding dropout;
- three-state first/all-caps embeddings versus four-state mixed-case embeddings;
- with versus without the auxiliary capitalization prediction loss;
- no-capitalization-embedding control initialized from the matched uncased
  continuation.

These ablations should be run on CoNLL and Walia first, because they directly
test the token/entity claim.

## 8. Error Analysis

The planned error-analysis tool is `scripts/error_analysis_by_case.py`. It
groups token-level correctness by source capitalization class. This will allow
the paper to report whether gains concentrate on first-cap entities, all-caps
acronyms, or mixed-case forms.

The analysis should be run once the prediction JSONL roots are mounted again
from the RunPod network volume.

## 9. Limitations

Capitalization embeddings are not a universal replacement for cased
tokenization. TweetEval Emoji is a clear negative control: cased BERT strongly
outperforms both uncased and capitalization-embedding models. TREC Fine is
another miss, where the capitalized model underperforms both matched baselines.
Scientific relation and SciEntsBank diagnostics also do not support a broad
"best of both worlds" claim.

These limitations shape the correct claim. The method is strongest when
capitalization behaves as a reusable feature over shared lexical items,
especially in token/entity tasks. It is less reliable when capitalization is
part of a task-specific social signal or when cased tokenization provides
broader lexical advantages that are not captured by a small case channel.

## 10. Conclusion

Capitalization embeddings factor case away from lexical identity in BERT-style
models. The method adds negligible downstream parameters, preserves uncased
lexical sharing, and recovers much of the cased-model signal on token/entity
benchmarks. The evidence supports a targeted architectural claim, not universal
dominance. With ablations and error analysis added, the work can be framed as a
practical and interpretable alternative to vocabulary-level casing decisions.

## Current Citation Map

The starter bibliography is in `paper/references.bib`. The current draft should
use the following citation map when converted to LaTeX:

- BERT: `devlin-etal-2019-bert`
- Hugging Face implementation stack: `wolf-etal-2020-transformers`,
  `lhoest-etal-2021-datasets`
- WordPiece/subword background: `schuster-nakajima-2012-voice`,
  `wu-etal-2016-gnmt`
- Token/entity datasets: `tjong-kim-sang-de-meulder-2003-introduction`,
  `derczynski-etal-2017-results`, `weischedel-etal-2013-ontonotes`,
  `marcus-etal-1993-penn-treebank`
- Sequence/control datasets: `barbieri-etal-2020-tweeteval`,
  `socher-etal-2013-recursive`, `lang-1995-newsweeder`,
  `li-roth-2002-learning`, `abu-farha-etal-2022-semeval`
- Supporting and appendix diagnostics: `rjac-2022-kaggle-entity-annotated-corpus`,
  `bos-etal-2017-groningen`, `gabor-etal-2018-semeval`,
  `luan-etal-2018-multi`, `dzikovska-etal-2013-semeval`,
  `athar-2011-sentiment`
- Statistical testing: `koehn-2004-statistical`, `holm-1979-simple`

Open citation issues are tracked in `CITATION_STATUS.md`.
