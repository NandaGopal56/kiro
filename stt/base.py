"""Abstract base for speech-to-text providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class STTProvider(ABC):
    """Common interface all STT providers must implement."""

    @abstractmethod
    async def stream(self) -> AsyncIterator[str]:
        """Yield transcript strings as they become available."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources (sockets, threads, mic streams)."""
        ...