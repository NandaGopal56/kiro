from __future__ import annotations

from . import chunkers, embedders, loaders, stores, summarizers
from .chunkers.base import Chunker
from .config import RAGConfig
from .loaders.base import Loader
from .stores.base import VectorStore
from .types import Doc, ParsedDoc


class Ingestor:
    """Run a single source through load -> chunk -> store.

    Compose by hand or via Ingestor.from_config(RAGConfig(...)).
    """

    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        store: VectorStore | None = None,
        include_images: bool = True,
        include_tables: bool = True,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.store = store
        self.include_images = include_images
        self.include_tables = include_tables

    @classmethod
    def from_config(cls, config: RAGConfig | None = None) -> "Ingestor":
        config = config or RAGConfig()

        loader_kwargs = dict(config.loader_kwargs)
        if config.summarizer and "summarizer" not in loader_kwargs:
            loader_kwargs["summarizer"] = summarizers.get(
                config.summarizer, **config.summarizer_kwargs
            )

        store_kwargs = dict(config.store_kwargs)
        if "embedder" not in store_kwargs:
            store_kwargs["embedder"] = embedders.get(
                config.embedder, **config.embedder_kwargs
            )

        return cls(
            loader=loaders.get(config.loader, **loader_kwargs),
            chunker=chunkers.get(config.chunker, **config.chunker_kwargs),
            store=stores.get(config.store, **store_kwargs),
            include_images=config.include_images,
            include_tables=config.include_tables,
        )

    def parse(self, source: str) -> ParsedDoc:
        return self.loader.load(source)

    def chunk(self, parsed: ParsedDoc) -> list[Doc]:
        texts = parsed.all_texts(
            include_images=self.include_images,
            include_tables=self.include_tables,
        )
        return self.chunker.split(texts)

    def run(self, source: str) -> list[Doc]:
        return self.chunk(self.parse(source))

    def ingest(self, source: str, upsert: bool = True) -> list[str]:
        if self.store is None:
            raise ValueError("No store configured for this ingestor.")
        docs = self.run(source)
        if upsert:
            return self.store.upsert_documents(docs)
        return self.store.add_documents(docs)

    def search(self, query: str, k: int = 4) -> list[Doc]:
        if self.store is None:
            raise ValueError("No store configured for this ingestor.")
        return self.store.similarity_search(query, k=k)
