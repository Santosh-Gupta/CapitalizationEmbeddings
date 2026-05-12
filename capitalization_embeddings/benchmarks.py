"""Benchmark registry for capitalization-sensitive downstream tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata for a candidate capitalization-sensitive benchmark."""

    key: str
    task_type: str
    dataset_name: str
    dataset_config: str | None
    text_columns: tuple[str, ...]
    label_column: str
    metric: str
    priority: int
    why_capitalization_matters: str
    status: str = "candidate"


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="conll2003_ner",
        task_type="token_classification",
        dataset_name="lhoestq/conll2003",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="ner_tags",
        metric="seqeval_f1",
        priority=1,
        why_capitalization_matters=(
            "Newswire named entities strongly depend on case, especially person, "
            "organization, location, and miscellaneous entity boundaries."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="wnut17_ner",
        task_type="token_classification",
        dataset_name="flaitenberger/wnut_17",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="ner_tags",
        metric="seqeval_f1",
        priority=2,
        why_capitalization_matters=(
            "Emerging and noisy social-media entities test whether capitalization "
            "features help outside clean newswire text."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="ontonotes5_ner",
        task_type="token_classification",
        dataset_name="extraordinarylab/ontonotes5",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="ner_tags",
        metric="seqeval_f1",
        priority=3,
        why_capitalization_matters=(
            "Larger multi-genre NER benchmark with many proper-name and acronym "
            "categories; useful if dataset access and labels are stable."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="ptb_pos",
        task_type="token_classification",
        dataset_name="batterydata/pos_tagging",
        dataset_config=None,
        text_columns=("words",),
        label_column="labels",
        metric="accuracy",
        priority=4,
        why_capitalization_matters=(
            "Penn Treebank-style POS tagging should benefit from case cues for "
            "proper nouns, sentence-initial ambiguity, and acronyms."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="conll2003_pos",
        task_type="token_classification",
        dataset_name="lhoestq/conll2003",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="pos_tags",
        metric="accuracy",
        priority=5,
        why_capitalization_matters=(
            "Proper-noun POS tags are case-sensitive, making this a cheap auxiliary "
            "check alongside CoNLL NER."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="tweet_eval_irony",
        task_type="sequence_classification",
        dataset_name="tweet_eval",
        dataset_config="irony",
        text_columns=("text",),
        label_column="label",
        metric="macro_f1",
        priority=10,
        why_capitalization_matters=(
            "Noisy social text where casing can be expressive or inconsistent; "
            "reported BERT baselines often favor uncased."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="tweet_eval_sentiment",
        task_type="sequence_classification",
        dataset_name="tweet_eval",
        dataset_config="sentiment",
        text_columns=("text",),
        label_column="label",
        metric="macro_f1",
        priority=11,
        why_capitalization_matters=(
            "Social sentiment classification where lexical unification may matter "
            "more than preserving case."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="tweet_eval_offensive",
        task_type="sequence_classification",
        dataset_name="tweet_eval",
        dataset_config="offensive",
        text_columns=("text",),
        label_column="label",
        metric="macro_f1",
        priority=12,
        why_capitalization_matters=(
            "Noisy social classification with inconsistent casing."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="tweet_eval_emotion",
        task_type="sequence_classification",
        dataset_name="tweet_eval",
        dataset_config="emotion",
        text_columns=("text",),
        label_column="label",
        metric="macro_f1",
        priority=13,
        why_capitalization_matters=(
            "Noisy emotion classification where uncased baselines can be stronger."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="sst5",
        task_type="sequence_classification",
        dataset_name="SetFit/sst5",
        dataset_config=None,
        text_columns=("text",),
        label_column="label",
        metric="accuracy",
        priority=14,
        why_capitalization_matters=(
            "Fine-grained sentiment where case is usually not the core signal."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="stsb",
        task_type="sequence_regression",
        dataset_name="glue",
        dataset_config="stsb",
        text_columns=("sentence1", "sentence2"),
        label_column="label",
        metric="pearson",
        priority=15,
        why_capitalization_matters=(
            "Semantic similarity should reward lexical sharing across casing."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="twenty_newsgroups",
        task_type="sequence_classification",
        dataset_name="SetFit/20_newsgroups",
        dataset_config=None,
        text_columns=("text",),
        label_column="label",
        metric="accuracy",
        priority=16,
        why_capitalization_matters=(
            "Topic classification where vocabulary sharing can outweigh case cues."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="yahoo_answers_topics",
        task_type="sequence_classification",
        dataset_name="yahoo_answers_topics",
        dataset_config=None,
        text_columns=("question_title", "question_content", "best_answer"),
        label_column="topic",
        metric="accuracy",
        priority=17,
        why_capitalization_matters=(
            "Noisy web topic classification where uncased BERT can have an edge."
        ),
        status="implemented",
    ),
)


def benchmark_keys() -> list[str]:
    """Return registered benchmark keys in priority order."""

    return [spec.key for spec in sorted(BENCHMARKS, key=lambda item: item.priority)]


def get_benchmark(key: str) -> BenchmarkSpec:
    """Return a benchmark spec by key."""

    for spec in BENCHMARKS:
        if spec.key == key:
            return spec
    available = ", ".join(benchmark_keys())
    raise KeyError(f"Unknown benchmark {key!r}. Available benchmarks: {available}")
