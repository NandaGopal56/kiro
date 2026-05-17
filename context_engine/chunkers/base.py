from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..text import clean_chunk
from ..types import Doc


class Chunker(ABC):
    """Base for text chunking strategies."""

    name = "base"

    def __init__(
        self,
        source: str = "document",
        clean_output: bool = True,
        include_empty: bool = False,
    ) -> None:
        self.source = source
        self.clean_output = clean_output
        self.include_empty = include_empty

    @abstractmethod
    def split(self, texts: list[str]) -> list[Doc]:
        """Split input texts into Doc chunks."""

    def _finalize(self, chunks: Iterable[str]) -> list[Doc]:
        docs: list[Doc] = []
        for raw in chunks:
            text = clean_chunk(raw) if self.clean_output else raw.strip()
            if not text and not self.include_empty:
                continue
            docs.append(
                Doc(
                    page_content=text,
                    metadata={
                        "chunk_id": len(docs),
                        "source": self.source,
                        "chunker": self.name,
                    },
                )
            )
        return docs
