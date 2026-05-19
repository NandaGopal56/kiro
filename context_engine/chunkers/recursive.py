from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schema import Doc
from .base import Chunker


class RecursiveChunker(Chunker):
    """Recursive character chunking with language-aware separators."""

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def split(self, texts: list[str]) -> list[Doc]:
        chunks = self.splitter.create_documents(texts)
        return self._finalize(chunk.page_content for chunk in chunks)
