from __future__ import annotations

from typing import Any

from .base import Chunker
from .fixed import FixedChunker
from .markdown import MarkdownChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

REGISTRY: dict[str, type[Chunker]] = {
    "fixed": FixedChunker,
    "character": FixedChunker,
    "markdown": MarkdownChunker,
    "md": MarkdownChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get(name: str, **kwargs: Any) -> Chunker:
    try:
        cls = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown chunker {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc
    return cls(**kwargs)


__all__ = [
    "Chunker",
    "FixedChunker",
    "MarkdownChunker",
    "REGISTRY",
    "RecursiveChunker",
    "SemanticChunker",
    "get",
]
