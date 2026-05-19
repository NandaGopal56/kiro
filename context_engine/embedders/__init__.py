from __future__ import annotations

from typing import Any

from .base import Embedder
from .openai import OpenAIEmbedder

REGISTRY: dict[str, type] = {
    "openai": OpenAIEmbedder,
}


def get(name: str, **kwargs: Any) -> Embedder:
    try:
        cls = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown embedder {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc
    return cls(**kwargs)


__all__ = ["Embedder", "OpenAIEmbedder", "REGISTRY", "get"]
