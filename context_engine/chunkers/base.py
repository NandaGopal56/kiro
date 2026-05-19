from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable

from .schema import Doc


def clean_chunk(text: str) -> str:
    """Normalize text at the chunk level."""

    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"Print to PDF", "", text)
    text = re.sub(r"^\s*\.\s*", "", text)
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"and drying, green coffee beans.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    async def split(self, texts: list[str]) -> list[Doc]:
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
