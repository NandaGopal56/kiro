from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RAGConfig:
    """Config for composing an Ingestor by component name.

    Add a new source/provider by registering it in the matching subpackage and
    flipping the corresponding name here.
    """

    loader: str = "pdf"
    loader_kwargs: dict[str, Any] = field(default_factory=dict)

    chunker: str = "recursive"
    chunker_kwargs: dict[str, Any] = field(default_factory=dict)

    embedder: str = "hash"
    embedder_kwargs: dict[str, Any] = field(default_factory=dict)

    store: str = "chroma"
    store_kwargs: dict[str, Any] = field(default_factory=dict)

    summarizer: str | None = None
    summarizer_kwargs: dict[str, Any] = field(default_factory=dict)

    include_images: bool = True
    include_tables: bool = True
