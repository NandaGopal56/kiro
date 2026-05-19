from __future__ import annotations

from langchain_experimental.text_splitter import SemanticChunker as _SemanticChunker
from langchain_openai import OpenAIEmbeddings

from .schema import Doc
from .base import Chunker


class SemanticChunker(Chunker):
    """Embedding-driven semantic chunking."""

    name = "semantic"

    def __init__(
        self,
        embeddings=None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float | None = None,
        min_chunk_size: int | None = 300,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        embeddings = embeddings or OpenAIEmbeddings()
        splitter_kwargs = {
            "embeddings": embeddings,
            "breakpoint_threshold_type": breakpoint_threshold_type,
            "min_chunk_size": min_chunk_size,
        }
        if breakpoint_threshold_amount is not None:
            splitter_kwargs["breakpoint_threshold_amount"] = breakpoint_threshold_amount
        self.splitter = _SemanticChunker(**splitter_kwargs)

    async def split(self, texts: list[str]) -> list[Doc]:
        chunks: list[str] = []
        for text in texts:
            chunks.extend(self.splitter.split_text(text))
        return self._finalize(chunks)
