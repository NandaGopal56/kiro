from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class VLMQuery:
    prompt: str
    frame=None
    history: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VLMResponse:
    text: str
    model: str
    raw: Optional[object] = None


class VisionLanguageModel(ABC):
    """Adapter interface for any vision-language model backend.

    Implementations turn a frame (image) plus a natural-language prompt
    into a textual answer. Concrete backends (OpenAI, local LLaVA, etc.)
    live behind this boundary so the rest of the system stays backend-agnostic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def analyze(self, frame, prompt: str, history: Optional[list[str]] = None) -> VLMResponse:
        ...

    def answer(self, query: VLMQuery) -> VLMResponse:
        return self.analyze(query.frame, query.prompt, query.history)
