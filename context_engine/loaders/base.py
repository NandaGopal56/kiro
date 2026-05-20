from __future__ import annotations

from abc import ABC, abstractmethod
from .schema import ParsedDoc


class Loader(ABC):
    """Base for source-specific loaders. Implementations turn a source into a ParsedDoc."""

    @abstractmethod
    async def load(self, source: str) -> ParsedDoc:
        ...
