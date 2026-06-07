from __future__ import annotations

from typing import Any, Type
from importlib import import_module

from .base import Loader


REGISTRY: dict[str, str] = {
    "auto": "context_engine.loaders.auto.AutoLoader",
    "pdf": "context_engine.loaders.pdf_markdown.PdfMarkdownLoader",
    "pdf_markdown": "context_engine.loaders.pdf_markdown.PdfMarkdownLoader",
}


def _load(path: str) -> Type[Loader]:
    module_path, cls_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, cls_name)


def get(name: str, **kwargs: Any) -> Loader:
    try:
        path = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown loader {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc

    cls = _load(path)
    return cls(**kwargs)


__all__ = ["Loader", "get"]