from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Answer:
    """RAG output shape used by evaluation adapters."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Sample:
    """One item to evaluate. Stays independent from any RAG implementation."""

    question: str
    expected_answer: str | None = None
    actual_answer: str | None = None
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Result:
    metric: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
