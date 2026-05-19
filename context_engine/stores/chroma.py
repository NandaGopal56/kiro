from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import chromadb

from ..embedders.base import Embedder
from ..embedders.openai import OpenAIEmbedder
from ..chunkers.schema import Doc


class ChromaStore:
    """CRUD and similarity-search operations for Docs backed by Chroma."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = ".rag_chroma",
        embedder: Embedder | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder = embedder or OpenAIEmbedder()
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_documents(
        self,
        docs: Sequence[Doc],
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        if not docs:
            return []
        ids = list(ids) if ids is not None else self._default_ids(docs)
        embeddings = await self.embedder.embed_documents([d.page_content for d in docs])
        self.collection.add(
            ids=ids,
            documents=[d.page_content for d in docs],
            metadatas=[self._metadata(d.metadata) for d in docs],
            embeddings=embeddings,
        )
        return ids

    async def upsert_documents(
        self,
        docs: Sequence[Doc],
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        if not docs:
            return []
        ids = list(ids) if ids is not None else self._default_ids(docs)
        embeddings = await self.embedder.embed_documents([d.page_content for d in docs])
        self.collection.upsert(
            ids=ids,
            documents=[d.page_content for d in docs],
            metadatas=[self._metadata(d.metadata) for d in docs],
            embeddings=embeddings,
        )
        return ids

    async def get_documents(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Doc]:
        result = self.collection.get(
            ids=list(ids) if ids is not None else None,
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        return self._to_docs(result)

    async def update_documents(
        self,
        docs: Sequence[Doc],
        ids: Sequence[str],
    ) -> None:
        if not docs:
            return
        embeddings = await self.embedder.embed_documents([d.page_content for d in docs])
        self.collection.update(
            ids=list(ids),
            documents=[d.page_content for d in docs],
            metadatas=[self._metadata(d.metadata) for d in docs],
            embeddings=embeddings,
        )

    async def delete_documents(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        self.collection.delete(
            ids=list(ids) if ids is not None else None, where=where
        )

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[Doc]:
        query_embedding = await self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            Doc(page_content=content, metadata={**(metadata or {}), "distance": distance})
            for content, metadata, distance in zip(
                documents, metadatas, distances, strict=False
            )
        ]

    async def count(self) -> int:
        return self.collection.count()

    async def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _default_ids(self, docs: Sequence[Doc]) -> list[str]:
        ids: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "document")
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is not None:
                ids.append(f"{source}:{chunk_id}")
                continue
            digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
            ids.append(f"{source}:{digest}")
        return ids

    def _metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        clean: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str | int | float | bool):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean

    def _to_docs(self, result: dict[str, Any]) -> list[Doc]:
        contents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            Doc(page_content=content, metadata=metadata or {})
            for content, metadata in zip(contents, metadatas, strict=False)
        ]


def _sample_docs() -> list[Doc]:
    return [
        Doc(
            page_content="Washed coffee processing uses water to remove fruit before drying.",
            metadata={"source": "sample", "chunk_id": 0},
        ),
        Doc(
            page_content="Natural coffee processing dries the coffee cherry before hulling.",
            metadata={"source": "sample", "chunk_id": 1},
        ),
        Doc(
            page_content="Honey processing leaves some mucilage on coffee beans while drying.",
            metadata={"source": "sample", "chunk_id": 2},
        ),
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test Chroma store operations.")
    parser.add_argument("--db-path", default=".rag_chroma_test")
    parser.add_argument("--collection", default="rag_smoke_test")
    parser.add_argument("--query", default="Which coffee process uses water?")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    store = ChromaStore(
        collection_name=args.collection,
        persist_directory=args.db_path,
        embedder=HashEmbedder(),
    )
    if args.reset:
        store.clear()

    ids = store.upsert_documents(_sample_docs())
    matches = store.similarity_search(args.query, k=2)

    print(f"Vector DB path: {Path(args.db_path).resolve()}")
    print(f"Collection: {args.collection}")
    print(f"Upserted ids: {ids}")
    print(f"Total documents: {store.count()}")
    print("Top matches:")
    for index, match in enumerate(matches, start=1):
        print(f"{index}. distance={match.metadata['distance']:.4f} {match.page_content}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
