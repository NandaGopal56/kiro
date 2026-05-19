from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..chunkers.schema import Doc


class VectorStore(Protocol):
    """Contract that any vector store implementation must satisfy."""

    async def add_documents(
        self, docs: Sequence[Doc], ids: Sequence[str] | None = ...
    ) -> list[str]: ...

    async def upsert_documents(
        self, docs: Sequence[Doc], ids: Sequence[str] | None = ...
    ) -> list[str]: ...

    async def get_documents(
        self,
        ids: Sequence[str] | None = ...,
        where: dict[str, Any] | None = ...,
        limit: int | None = ...,
    ) -> list[Doc]: ...

    async def update_documents(self, docs: Sequence[Doc], ids: Sequence[str]) -> None: ...

    async def delete_documents(
        self,
        ids: Sequence[str] | None = ...,
        where: dict[str, Any] | None = ...,
    ) -> None: ...

    async def similarity_search(
        self,
        query: str,
        k: int = ...,
        where: dict[str, Any] | None = ...,
    ) -> list[Doc]: ...

    async def count(self) -> int: ...

    async def clear(self) -> None: ...
