from __future__ import annotations

from typing import Any

from .auto import AutoLoader
from .base import ExtractionResult, Loader
from .pdf import PdfLoader
from .pdf_markdown import PdfMarkdownLoader

REGISTRY: dict[str, type[Loader]] = {
    "auto": AutoLoader,
    "pdf": PdfLoader,
    "pdf_markdown": PdfMarkdownLoader,
}


def get(name: str, **kwargs: Any) -> Loader:
    try:
        cls = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown loader {name!r}. Available: {sorted(REGISTRY)}"
        ) from exc
    return cls(**kwargs)


__all__ = [
    "AutoLoader",
    "ExtractionResult",
    "Loader",
    "PdfLoader",
    "PdfMarkdownLoader",
    "REGISTRY",
    "get",
]
