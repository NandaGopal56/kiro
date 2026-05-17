from __future__ import annotations

from typing import Any

from .base import Summarizer
from .openai import OpenAISummarizer

REGISTRY: dict[str, type] = {
    "openai": OpenAISummarizer,
}


def get(name: str, **kwargs: Any) -> Summarizer:
    try:
        cls = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown summarizer {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc
    return cls(**kwargs)


__all__ = ["OpenAISummarizer", "REGISTRY", "Summarizer", "get"]
