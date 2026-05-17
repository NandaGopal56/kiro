from __future__ import annotations

from langchain_text_splitters import CharacterTextSplitter

from ..engine import Doc
from .base import Chunker


class FixedChunker(Chunker):
    """Fixed-size character chunking."""

    name = "fixed"

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 0,
        separator: str = "\n\n",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, texts: list[str]) -> list[Doc]:
        chunks = self.splitter.create_documents(texts)
        return self._finalize(chunk.page_content for chunk in chunks)
