from __future__ import annotations

from typing import Any, Type
from importlib import import_module

from .base import Embedder


REGISTRY: dict[str, str] = {
    "openai": "context_engine.embedders.openai.OpenAIEmbedder",
}


def _load(path: str) -> Type[Embedder]:
    module_path, cls_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, cls_name)


def get(name: str, **kwargs: Any) -> Embedder:
    try:
        path = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown embedder {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc

    cls = _load(path)
    return cls(**kwargs)


__all__ = ["Embedder", "get"]