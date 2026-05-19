import pytest

from context_engine.evals.metrics import AnswerContains, ContextContains
from context_engine.evals.types import Sample


def test_answer_contains_name():
    metric = AnswerContains()
    assert metric.name == "answer_contains"


def test_answer_contains_pass():
    sample = Sample(
        question="Test",
        expected_answer="hello world",
        actual_answer="The answer is HELLO WORLD"
    )
    metric = AnswerContains()
    result = metric.evaluate(sample)
    assert result.passed is True
    assert result.score == 1.0


def test_answer_contains_fail():
    sample = Sample(
        question="Test",
        expected_answer="hello world",
        actual_answer="The answer is goodbye"
    )
    metric = AnswerContains()
    result = metric.evaluate(sample)
    assert result.passed is False
    assert result.score == 0.0


def test_answer_contains_none_expected():
    sample = Sample(
        question="Test",
        expected_answer=None,
        actual_answer="Some answer"
    )
    metric = AnswerContains()
    result = metric.evaluate(sample)
    assert result.passed is False


def test_context_contains_name():
    metric = ContextContains()
    assert metric.name == "context_contains"


def test_context_contains_pass():
    sample = Sample(
        question="Test",
        expected_answer="hello world",
        contexts=["The context contains HELLO WORLD information"]
    )
    metric = ContextContains()
    result = metric.evaluate(sample)
    assert result.passed is True
    assert result.score == 1.0


def test_context_contains_fail():
    sample = Sample(
        question="Test",
        expected_answer="hello world",
        contexts=["The context contains other information"]
    )
    metric = ContextContains()
    result = metric.evaluate(sample)
    assert result.passed is False
    assert result.score == 0.0
