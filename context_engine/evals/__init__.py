"""Evaluation primitives that measure RAG without depending on it."""

from .metrics import AnswerContains, ContextContains, Metric
from .runner import RAGCallable, evaluate, run
from .types import Answer, Result, Sample

__all__ = [
    "Answer",
    "AnswerContains",
    "ContextContains",
    "Metric",
    "RAGCallable",
    "Result",
    "Sample",
    "evaluate",
    "run",
]
