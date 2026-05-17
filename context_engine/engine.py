from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import chunkers, embedders, loaders, stores, summarizers
from .chunkers.base import Chunker
from .config import ContextEngineConfig
from .documents import Doc, ParsedDoc
from .loaders.base import Loader
from .stores.base import VectorStore


class ContextEngine:
    """Client-facing entry point for parsing, chunking, ingestion, and retrieval."""

    def __init__(
        self,
        config: ContextEngineConfig | None = None,
        *,
        loader: Loader | None = None,
        chunker: Chunker | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.config = config or ContextEngineConfig()
        self.loader = loader or self._make_loader(self.config.loader)
        self.chunker = chunker or self._make_chunker(self.config.chunker)
        self._store = store

    def parse(
        self,
        source: str,
        *,
        loader: str | Loader | None = None,
        loader_kwargs: dict[str, Any] | None = None,
    ) -> ParsedDoc:
        return self._loader(loader, loader_kwargs).load(source)

    def chunk(
        self,
        parsed: ParsedDoc | list[str] | str,
        *,
        chunker: str | Chunker | None = None,
        chunker_kwargs: dict[str, Any] | None = None,
        include_images: bool | None = None,
        include_tables: bool | None = None,
    ) -> list[Doc]:
        texts = self._texts(parsed, include_images, include_tables)
        return self._chunker(chunker, chunker_kwargs).split(texts)

    def run(
        self,
        source: str,
        *,
        loader: str | Loader | None = None,
        chunker: str | Chunker | None = None,
        loader_kwargs: dict[str, Any] | None = None,
        chunker_kwargs: dict[str, Any] | None = None,
        include_images: bool | None = None,
        include_tables: bool | None = None,
    ) -> list[Doc]:
        parsed = self.parse(source, loader=loader, loader_kwargs=loader_kwargs)
        return self.chunk(
            parsed,
            chunker=chunker,
            chunker_kwargs=chunker_kwargs,
            include_images=include_images,
            include_tables=include_tables,
        )

    def ingest(
        self,
        source: str,
        *,
        loader: str | Loader | None = None,
        chunker: str | Chunker | None = None,
        store: str | VectorStore | None = None,
        loader_kwargs: dict[str, Any] | None = None,
        chunker_kwargs: dict[str, Any] | None = None,
        store_kwargs: dict[str, Any] | None = None,
        include_images: bool | None = None,
        include_tables: bool | None = None,
        upsert: bool = True,
    ) -> list[str]:
        docs = self.run(
            source,
            loader=loader,
            chunker=chunker,
            loader_kwargs=loader_kwargs,
            chunker_kwargs=chunker_kwargs,
            include_images=include_images,
            include_tables=include_tables,
        )
        target_store = self._store_for(store, store_kwargs)
        if upsert:
            return target_store.upsert_documents(docs)
        return target_store.add_documents(docs)

    def retrieve(
        self,
        query: str,
        *,
        k: int = 4,
        store: str | VectorStore | None = None,
        store_kwargs: dict[str, Any] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Doc]:
        return self._store_for(store, store_kwargs).similarity_search(
            query,
            k=k,
            where=where,
        )

    def search(self, query: str, k: int = 4) -> list[Doc]:
        return self.retrieve(query, k=k)

    def count(self) -> int:
        return self.store.count()

    def clear(self) -> None:
        self.store.clear()

    def tools(self) -> dict[str, Callable[..., Any]]:
        return {
            "context_ingest": self.ingest,
            "context_retrieve": self.retrieve,
            "context_parse": self.parse,
            "context_chunk": self.chunk,
        }

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._make_store(self.config.store)
        return self._store

    def _loader(
        self,
        loader: str | Loader | None,
        overrides: dict[str, Any] | None,
    ) -> Loader:
        if loader is None and overrides is None:
            return self.loader
        if isinstance(loader, Loader):
            return loader
        return self._make_loader(loader or self.config.loader, overrides)

    def _chunker(
        self,
        chunker: str | Chunker | None,
        overrides: dict[str, Any] | None,
    ) -> Chunker:
        if chunker is None and overrides is None:
            return self.chunker
        if isinstance(chunker, Chunker):
            return chunker
        return self._make_chunker(chunker or self.config.chunker, overrides)

    def _store_for(
        self,
        store: str | VectorStore | None,
        overrides: dict[str, Any] | None,
    ) -> VectorStore:
        if store is None and overrides is None:
            return self.store
        if not isinstance(store, str) and store is not None:
            return store
        return self._make_store(store or self.config.store, overrides)

    def _make_loader(
        self,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> Loader:
        kwargs = {**self.config.loader_kwargs, **(overrides or {})}
        if self.config.summarizer and "summarizer" not in kwargs:
            kwargs["summarizer"] = summarizers.get(
                self.config.summarizer,
                **self.config.summarizer_kwargs,
            )
        return loaders.get(name, **kwargs)

    def _make_chunker(
        self,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> Chunker:
        kwargs = {**self.config.chunker_kwargs, **(overrides or {})}
        return chunkers.get(name, **kwargs)

    def _make_store(
        self,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> VectorStore:
        kwargs = {**self.config.store_kwargs, **(overrides or {})}
        if "embedder" not in kwargs:
            kwargs["embedder"] = embedders.get(
                self.config.embedder,
                **self.config.embedder_kwargs,
            )
        return stores.get(name, **kwargs)

    def _texts(
        self,
        parsed: ParsedDoc | list[str] | str,
        include_images: bool | None,
        include_tables: bool | None,
    ) -> list[str]:
        if isinstance(parsed, str):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return parsed.all_texts(
            include_images=self.config.include_images
            if include_images is None
            else include_images,
            include_tables=self.config.include_tables
            if include_tables is None
            else include_tables,
        )
