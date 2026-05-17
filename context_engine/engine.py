from __future__ import annotations

from typing import Any

from . import chunkers, embedders, loaders, stores
from .chunkers.base import Chunker
from .chunkers.schema import Doc
from .loaders.base import Loader
from .stores.base import VectorStore


class ContextEngine:
    """Client-facing entry point for parsing, chunking, ingestion, and retrieval.
    
    Components are fixed internally. For customization, instantiate components
    separately and pass them to the constructor.
    """

    def __init__(
        self,
        loader: Loader | None = None,
        chunker: Chunker | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.loader = loader or loaders.get("auto")
        self.chunker = chunker or chunkers.get("recursive")
        self._store = store

    def ingest(
        self,
        source: str,
        include_images: bool = True,
        include_tables: bool = True,
        upsert: bool = True,
    ) -> list[str]:
        parsed = self.loader.load(source)
        texts = parsed.all_texts(
            include_images=include_images, 
            include_tables=include_tables
        )
        docs = self.chunker.split(texts)
        if upsert:
            return self.store.upsert_documents(docs)
        return self.store.add_documents(docs)

    def retrieve(
        self,
        query: str,
        k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[Doc]:
        return self.store.similarity_search(
            query,
            k=k,
            where=where,
        )

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = stores.get("chroma", embedder=embedders.get("hash"))
        return self._store
