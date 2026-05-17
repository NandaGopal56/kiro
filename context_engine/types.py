from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Doc:
    """A single chunk-shaped document used by chunkers, stores, and retrievers."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}


@dataclass(slots=True)
class ParsedDoc:
    """Loader output: normalized text/image/table strings before chunking."""

    texts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {"texts": self.texts, "images": self.images, "tables": self.tables}

    def all_texts(self, include_images: bool = True, include_tables: bool = True) -> list[str]:
        out = list(self.texts)
        if include_images:
            out.extend(self.images)
        if include_tables:
            out.extend(self.tables)
        return out
