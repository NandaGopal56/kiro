from __future__ import annotations

from typing import Any, Type
from importlib import import_module

from .base import Chunker


REGISTRY: dict[str, str] = {
    "fixed": "context_engine.chunkers.fixed.FixedChunker",
    "character": "context_engine.chunkers.fixed.FixedChunker",
    "markdown": "context_engine.chunkers.markdown.MarkdownChunker",
    "md": "context_engine.chunkers.markdown.MarkdownChunker",
    "recursive": "context_engine.chunkers.recursive.RecursiveChunker",
    "semantic": "context_engine.chunkers.semantic.SemanticChunker",
}


def _load(path: str) -> Type[Chunker]:
    module_path, cls_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, cls_name)


def get(name: str, **kwargs: Any) -> Chunker:
    try:
        path = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown chunker {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc

    cls = _load(path)
    return cls(**kwargs)


__all__ = ["Chunker", "get"]