from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Contract that any embedding provider must satisfy."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
