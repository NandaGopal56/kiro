from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Doc:
    """A chunk-shaped document used by chunkers, stores, and retrievers."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}


@dataclass(slots=True)
class ParsedDoc:
    """Normalized loader output before chunking."""

    texts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {"texts": self.texts, "images": self.images, "tables": self.tables}

    def all_texts(
        self,
        include_images: bool = True,
        include_tables: bool = True,
    ) -> list[str]:
        out = list(self.texts)
        if include_images:
            out.extend(self.images)
        if include_tables:
            out.extend(self.tables)
        return out


@dataclass(slots=True)
class ContextEngineConfig:
    """Default component choices for ContextEngine."""

    loader: str = "auto"
    loader_kwargs: dict[str, Any] = field(default_factory=dict)

    chunker: str = "recursive"
    chunker_kwargs: dict[str, Any] = field(default_factory=dict)

    embedder: str = "hash"
    embedder_kwargs: dict[str, Any] = field(default_factory=dict)

    store: str = "chroma"
    store_kwargs: dict[str, Any] = field(default_factory=dict)

    summarizer: str | None = None
    summarizer_kwargs: dict[str, Any] = field(default_factory=dict)

    include_images: bool = True
    include_tables: bool = True


# Registries import Doc/ParsedDoc from this module, so keep these imports after
# the small public data types above.
from . import chunkers, embedders, loaders, stores, summarizers
from .chunkers.base import Chunker
from .loaders.base import Loader
from .stores.base import VectorStore


class ContextEngine:
    """Small entry point for ingestion and retrieval.

    The constructor sets defaults for an app. Each method also accepts strategy
    overrides so agents and web handlers can select a process per request.
    """

    def __init__(
        self,
        config: ContextEngineConfig | None = None,
        *,
        loader: Loader | None = None,
        chunker: Chunker | None = None,
        store: VectorStore | None = None,
        include_images: bool | None = None,
        include_tables: bool | None = None,
    ) -> None:
        self.config = config or ContextEngineConfig()
        if include_images is not None:
            self.config.include_images = include_images
        if include_tables is not None:
            self.config.include_tables = include_tables
        self.loader = loader or self._build_loader()
        self.chunker = chunker or self._build_chunker()
        self._store = store

    @classmethod
    def from_config(cls, config: ContextEngineConfig | None = None) -> "ContextEngine":
        return cls(config=config)

    def parse(
        self,
        source: str,
        *,
        loader: str | Loader | None = None,
        loader_kwargs: dict[str, Any] | None = None,
    ) -> ParsedDoc:
        return self._resolve_loader(loader, loader_kwargs).load(source)

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
        return self._resolve_chunker(chunker, chunker_kwargs).split(texts)

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
        target_store = self._resolve_store(store, store_kwargs)
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
        return self._resolve_store(store, store_kwargs).similarity_search(
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

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._build_store()
        return self._store

    def tools(self) -> dict[str, Callable[..., Any]]:
        """Return framework-neutral callables for agent adapters."""

        return {
            "context_ingest": self.ingest,
            "context_retrieve": self.retrieve,
            "context_parse": self.parse,
            "context_chunk": self.chunk,
        }

    def _build_loader(self) -> Loader:
        return loaders.get(self.config.loader, **self._loader_kwargs())

    def _build_chunker(self) -> Chunker:
        return chunkers.get(self.config.chunker, **self._chunker_kwargs())

    def _build_store(self) -> VectorStore:
        store_kwargs = dict(self.config.store_kwargs)
        if "embedder" not in store_kwargs:
            store_kwargs["embedder"] = embedders.get(
                self.config.embedder,
                **self.config.embedder_kwargs,
            )
        return stores.get(self.config.store, **store_kwargs)

    def _resolve_loader(
        self,
        loader: str | Loader | None,
        loader_kwargs: dict[str, Any] | None,
    ) -> Loader:
        if loader is None and loader_kwargs is None:
            return self.loader
        if loader is None:
            return loaders.get(self.config.loader, **self._loader_kwargs(loader_kwargs))
        if isinstance(loader, str):
            return loaders.get(loader, **self._loader_kwargs(loader_kwargs))
        return loader

    def _resolve_chunker(
        self,
        chunker: str | Chunker | None,
        chunker_kwargs: dict[str, Any] | None,
    ) -> Chunker:
        if chunker is None and chunker_kwargs is None:
            return self.chunker
        if chunker is None:
            return chunkers.get(
                self.config.chunker,
                **self._chunker_kwargs(chunker_kwargs),
            )
        if isinstance(chunker, str):
            return chunkers.get(chunker, **self._chunker_kwargs(chunker_kwargs))
        return chunker

    def _loader_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = dict(self.config.loader_kwargs)
        kwargs.update(overrides or {})
        if self.config.summarizer and "summarizer" not in kwargs:
            kwargs["summarizer"] = summarizers.get(
                self.config.summarizer,
                **self.config.summarizer_kwargs,
            )
        return kwargs

    def _chunker_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = dict(self.config.chunker_kwargs)
        kwargs.update(overrides or {})
        return kwargs

    def _resolve_store(
        self,
        store: str | VectorStore | None,
        store_kwargs: dict[str, Any] | None,
    ) -> VectorStore:
        if store is None and store_kwargs is None:
            return self.store
        if store is None:
            return stores.get(self.config.store, **self._store_kwargs(store_kwargs))
        if isinstance(store, str):
            return stores.get(store, **self._store_kwargs(store_kwargs))
        return store

    def _store_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = dict(self.config.store_kwargs)
        kwargs.update(overrides or {})
        if "embedder" not in kwargs:
            kwargs["embedder"] = embedders.get(
                self.config.embedder,
                **self.config.embedder_kwargs,
            )
        return kwargs

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
            include_images=(
                self.config.include_images if include_images is None else include_images
            ),
            include_tables=(
                self.config.include_tables if include_tables is None else include_tables
            ),
        )
