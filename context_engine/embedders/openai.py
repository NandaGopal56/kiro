from __future__ import annotations

from typing import Any

from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedder:
    """OpenAI embeddings for production use."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            **kwargs,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(text)
