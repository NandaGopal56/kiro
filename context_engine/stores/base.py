from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..chunkers.schema import Doc


class VectorStore(Protocol):
    """Contract that any vector store implementation must satisfy."""

    def add_documents(
        self, docs: Sequence[Doc], ids: Sequence[str] | None = ...
    ) -> list[str]: ...

    def upsert_documents(
        self, docs: Sequence[Doc], ids: Sequence[str] | None = ...
    ) -> list[str]: ...

    def get_documents(
        self,
        ids: Sequence[str] | None = ...,
        where: dict[str, Any] | None = ...,
        limit: int | None = ...,
    ) -> list[Doc]: ...

    def update_documents(self, docs: Sequence[Doc], ids: Sequence[str]) -> None: ...

    def delete_documents(
        self,
        ids: Sequence[str] | None = ...,
        where: dict[str, Any] | None = ...,
    ) -> None: ...

    def similarity_search(
        self,
        query: str,
        k: int = ...,
        where: dict[str, Any] | None = ...,
    ) -> list[Doc]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...
