from __future__ import annotations

from typing import Any

from .base import VectorStore
from .chroma import ChromaStore

REGISTRY: dict[str, type] = {
    "chroma": ChromaStore,
}


def get(name: str, **kwargs: Any) -> VectorStore:
    try:
        cls = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown store {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc
    return cls(**kwargs)


__all__ = ["ChromaStore", "REGISTRY", "VectorStore", "get"]
