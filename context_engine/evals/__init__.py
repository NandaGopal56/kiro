from __future__ import annotations

from importlib import import_module
from typing import Any

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


def __getattr__(name: str) -> Any:
    mapping = {
        "Metric": "context_engine.evals.metrics",
        "AnswerContains": "context_engine.evals.metrics",
        "ContextContains": "context_engine.evals.metrics",
        "RAGCallable": "context_engine.evals.runner",
        "evaluate": "context_engine.evals.runner",
        "run": "context_engine.evals.runner",
        "Answer": "context_engine.evals.types",
        "Result": "context_engine.evals.types",
        "Sample": "context_engine.evals.types",
    }

    if name not in mapping:
        raise AttributeError(name)

    module = import_module(mapping[name])
    return getattr(module, name)