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
    processor: str | None = None


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
        key="tweet_eval_emoji",
        task_type="sequence_classification",
        dataset_name="tweet_eval",
        dataset_config="emoji",
        text_columns=("text",),
        label_column="label",
        metric="accuracy",
        priority=6,
        why_capitalization_matters=(
            "Social emoji prediction has one of the largest reported cased-over-"
            "uncased gaps in the IBM 36-task table; useful as a cased-favored "
            "sequence-classification counterpoint."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="trec_fine",
        task_type="sequence_classification",
        dataset_name="lukasgarbas/trec",
        dataset_config=None,
        text_columns=("text",),
        label_column="fine_label",
        metric="accuracy",
        priority=7,
        why_capitalization_matters=(
            "Fine-grained question classification is a standard compact task "
            "where cased BERT has a reported multi-point advantage over uncased."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="kaggle_walia_ner",
        task_type="token_classification",
        dataset_name="rjac/kaggle-entity-annotated-corpus-ner-dataset",
        dataset_config=None,
        text_columns=("tokens",),
        label_column="ner_tags",
        metric="seqeval_f1",
        priority=8,
        why_capitalization_matters=(
            "Kaggle/Walia NER is entity-heavy and has a reported cased-over-"
            "uncased gap, but the published comparison used BERT embeddings in "
            "another architecture, so treat it as supporting evidence."
        ),
        status="implemented",
        processor="single_train_token_split",
    ),
    BenchmarkSpec(
        key="isarcasm_eval_en",
        task_type="sequence_classification",
        dataset_name="iabufarha/iSarcasmEval",
        dataset_config=None,
        text_columns=("text",),
        label_column="label",
        metric="macro_f1",
        priority=9,
        why_capitalization_matters=(
            "Sarcasm in social text can use casing as an expressive cue; the "
            "original iSarcasmEval English task is a small but relevant "
            "cased-favored candidate."
        ),
        status="implemented",
        processor="isarcasm_eval_en_task_a",
    ),
    BenchmarkSpec(
        key="citation_sentiment_acl",
        task_type="sequence_classification",
        dataset_name="gaof23/citation_sentiment_corpus",
        dataset_config=None,
        text_columns=("text",),
        label_column="label",
        metric="macro_f1",
        priority=10,
        why_capitalization_matters=(
            "Citation sentiment is scientific text with named methods and "
            "acronyms. This is the public ACL citation sentiment corpus, not the "
            "tiny ACM 97-example test set, so reported ACM numbers should not be "
            "transferred without verification."
        ),
        status="implemented",
        processor="citation_sentiment_acl",
    ),
    BenchmarkSpec(
        key="tweet_eval_irony",
        task_type="sequence_classification",
        dataset_name="tweet_eval",
        dataset_config="irony",
        text_columns=("text",),
        label_column="label",
        metric="macro_f1",
        priority=11,
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
        priority=12,
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
        priority=13,
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
        priority=14,
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
        priority=15,
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
        priority=16,
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
        priority=17,
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
        priority=18,
        why_capitalization_matters=(
            "Noisy web topic classification where uncased BERT can have an edge."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="semeval2018_task7",
        task_type="sequence_classification",
        dataset_name="DFKI-SLT/SemEval2018_Task7",
        dataset_config="Subtask_1_1",
        text_columns=("text",),
        label_column="label",
        metric="accuracy",
        priority=19,
        why_capitalization_matters=(
            "Scientific relation classification where uncased baselines can be "
            "strong; entity and acronym casing should be preserved without "
            "fragmenting lexical evidence."
        ),
        status="implemented",
        processor="semeval2018_task7_relations",
    ),
    BenchmarkSpec(
        key="scierc_relations",
        task_type="sequence_classification",
        dataset_name="nsusemiehl/SciERC",
        dataset_config=None,
        text_columns=("text",),
        label_column="label",
        metric="accuracy",
        priority=20,
        why_capitalization_matters=(
            "Scientific relation classification with marked entity spans; useful "
            "for testing whether capitalization embeddings retain uncased-style "
            "lexical sharing in technical text."
        ),
        status="implemented",
    ),
    BenchmarkSpec(
        key="scientific_relations_combined",
        task_type="sequence_classification",
        dataset_name="combined:semeval2018_task7+scierc",
        dataset_config=None,
        text_columns=("text",),
        label_column="label",
        metric="accuracy",
        priority=21,
        why_capitalization_matters=(
            "Combined scientific relation classification benchmark, matching the "
            "reported setting where uncased BERT can outperform cased BERT."
        ),
        status="implemented",
        processor="combined_scientific_relations",
    ),
    BenchmarkSpec(
        key="scientbank_3way_uq",
        task_type="sequence_classification",
        dataset_name="nkazi/SciEntsBank",
        dataset_config=None,
        text_columns=("question", "reference_answer", "student_answer"),
        label_column="label",
        metric="macro_f1",
        priority=23,
        why_capitalization_matters=(
            "Automatic short-answer grading with unseen questions; reported "
            "uncased gains are large, likely because lexical sharing dominates "
            "over case-specific lexical entries."
        ),
        status="implemented",
        processor="scientbank_3way_uq",
    ),
    BenchmarkSpec(
        key="scientbank_3way_ud",
        task_type="sequence_classification",
        dataset_name="nkazi/SciEntsBank",
        dataset_config=None,
        text_columns=("question", "reference_answer", "student_answer"),
        label_column="label",
        metric="macro_f1",
        priority=22,
        why_capitalization_matters=(
            "Automatic short-answer grading with unseen domains; useful as a "
            "stress test for uncased-style lexical sharing under domain shift."
        ),
        status="implemented",
        processor="scientbank_3way_ud",
    ),
    BenchmarkSpec(
        key="scientbank_5way_uq",
        task_type="sequence_classification",
        dataset_name="nkazi/SciEntsBank",
        dataset_config=None,
        text_columns=("question", "reference_answer", "student_answer"),
        label_column="label",
        metric="accuracy",
        priority=24,
        why_capitalization_matters=(
            "Five-way short-answer grading with unseen questions; a related "
            "variant of SciEntsBank for checking whether the uncased advantage "
            "persists under a finer label taxonomy."
        ),
        status="implemented",
        processor="scientbank_5way_uq",
    ),
    BenchmarkSpec(
        key="scientbank_5way_ud",
        task_type="sequence_classification",
        dataset_name="nkazi/SciEntsBank",
        dataset_config=None,
        text_columns=("question", "reference_answer", "student_answer"),
        label_column="label",
        metric="accuracy",
        priority=25,
        why_capitalization_matters=(
            "Five-way short-answer grading with unseen domains; related to the "
            "3-way setting but useful as an additional robustness slice."
        ),
        status="implemented",
        processor="scientbank_5way_ud",
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
