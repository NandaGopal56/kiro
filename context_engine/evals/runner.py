from __future__ import annotations

from collections.abc import Callable, Iterable

from .metrics import Metric
from .types import Answer, Result, Sample

RAGCallable = Callable[[str], Answer]


def run(samples: Iterable[Sample], metrics: Iterable[Metric]) -> list[Result]:
    """Run independent metrics over independent samples."""

    metric_list = list(metrics)
    results: list[Result] = []
    for sample in samples:
        for metric in metric_list:
            results.append(metric.evaluate(sample))
    return results


def evaluate(
    rag: RAGCallable,
    samples: Iterable[Sample],
    metrics: Iterable[Metric],
) -> list[Result]:
    """Measure a RAG callable without coupling evaluation to the RAG pipeline."""

    measured: list[Sample] = []
    for sample in samples:
        answer = rag(sample.question)
        measured.append(
            Sample(
                question=sample.question,
                expected_answer=sample.expected_answer,
                actual_answer=answer.answer,
                contexts=answer.contexts,
                metadata={**sample.metadata, **answer.metadata},
            )
        )
    return run(measured, metrics)
