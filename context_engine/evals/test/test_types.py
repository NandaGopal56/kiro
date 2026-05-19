import pytest

from context_engine.evals.types import Answer, Sample, Result


def test_answer_creation():
    answer = Answer(answer="Test answer")
    assert answer.answer == "Test answer"
    assert answer.contexts == []
    assert answer.metadata == {}


def test_answer_with_contexts():
    answer = Answer(answer="Test", contexts=["context1", "context2"])
    assert answer.contexts == ["context1", "context2"]


def test_answer_with_metadata():
    answer = Answer(answer="Test", metadata={"key": "value"})
    assert answer.metadata == {"key": "value"}


def test_sample_creation():
    sample = Sample(question="What is test?")
    assert sample.question == "What is test?"
    assert sample.expected_answer is None
    assert sample.actual_answer is None
    assert sample.contexts == []


def test_sample_with_all_fields():
    sample = Sample(
        question="Test question",
        expected_answer="expected",
        actual_answer="actual",
        contexts=["ctx1"],
        metadata={"key": "value"}
    )
    assert sample.expected_answer == "expected"
    assert sample.actual_answer == "actual"


def test_result_creation():
    result = Result(metric="test_metric", score=1.0, passed=True)
    assert result.metric == "test_metric"
    assert result.score == 1.0
    assert result.passed is True
    assert result.details == {}


def test_result_with_details():
    result = Result(
        metric="test_metric",
        score=0.5,
        passed=False,
        details={"reason": "failed"}
    )
    assert result.details == {"reason": "failed"}
