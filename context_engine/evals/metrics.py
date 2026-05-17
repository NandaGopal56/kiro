from __future__ import annotations

from typing import Protocol

from .types import Result, Sample


class Metric(Protocol):
    """A metric receives one sample and returns one result."""

    name: str

    def evaluate(self, sample: Sample) -> Result: ...


def _contains(haystack: str, needle: str | None) -> bool:
    if not needle:
        return False
    return needle.casefold() in haystack.casefold()


class AnswerContains:
    """Simple deterministic answer check."""

    name = "answer_contains"

    def evaluate(self, sample: Sample) -> Result:
        passed = _contains(sample.actual_answer or "", sample.expected_answer)
        return Result(
            metric=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"expected_answer": sample.expected_answer},
        )


class ContextContains:
    """Simple deterministic retrieval-context check."""

    name = "context_contains"

    def evaluate(self, sample: Sample) -> Result:
        passed = _contains("\n".join(sample.contexts), sample.expected_answer)
        return Result(
            metric=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"context_count": len(sample.contexts)},
        )
