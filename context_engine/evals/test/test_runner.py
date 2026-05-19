import pytest

from context_engine.evals.metrics import AnswerContains
from context_engine.evals.runner import run, evaluate
from context_engine.evals.types import Answer, Sample


def test_run_with_single_sample():
    sample = Sample(
        question="Test",
        expected_answer="hello",
        actual_answer="HELLO world"
    )
    metric = AnswerContains()
    results = run([sample], [metric])
    assert len(results) == 1
    assert results[0].passed is True


def test_run_with_multiple_samples():
    samples = [
        Sample(question="Q1", expected_answer="hello", actual_answer="HELLO"),
        Sample(question="Q2", expected_answer="world", actual_answer="WORLD"),
    ]
    metric = AnswerContains()
    results = run(samples, [metric])
    assert len(results) == 2


def test_run_with_multiple_metrics():
    from context_engine.evals.metrics import ContextContains
    sample = Sample(
        question="Test",
        expected_answer="hello",
        actual_answer="HELLO",
        contexts=["HELLO context"]
    )
    metrics = [AnswerContains(), ContextContains()]
    results = run([sample], metrics)
    assert len(results) == 2


def test_evaluate_with_rag_callable():
    def mock_rag(question: str) -> Answer:
        return Answer(answer="HELLO", contexts=["HELLO context"])
    
    sample = Sample(question="Test", expected_answer="hello")
    metric = AnswerContains()
    results = evaluate(mock_rag, [sample], [metric])
    assert len(results) == 1
    assert results[0].passed is True


def test_evaluate_empty_samples():
    def mock_rag(question: str) -> Answer:
        return Answer(answer="test")
    
    metric = AnswerContains()
    results = evaluate(mock_rag, [], [metric])
    assert results == []
