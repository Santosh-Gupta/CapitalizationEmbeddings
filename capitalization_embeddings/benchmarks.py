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
        dataset_name="wnut_17",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="ner_tags",
        metric="seqeval_f1",
        priority=2,
        why_capitalization_matters=(
            "Emerging and noisy social-media entities test whether capitalization "
            "features help outside clean newswire text."
        ),
    ),
    BenchmarkSpec(
        key="ontonotes5_ner",
        task_type="token_classification",
        dataset_name="tner/ontonotes5",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="tags",
        metric="seqeval_f1",
        priority=3,
        why_capitalization_matters=(
            "Larger multi-genre NER benchmark with many proper-name and acronym "
            "categories; useful if dataset access and labels are stable."
        ),
    ),
    BenchmarkSpec(
        key="conll2003_pos",
        task_type="token_classification",
        dataset_name="lhoestq/conll2003",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="pos_tags",
        metric="accuracy",
        priority=4,
        why_capitalization_matters=(
            "Proper-noun POS tags are case-sensitive, making this a cheap auxiliary "
            "check alongside CoNLL NER."
        ),
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
