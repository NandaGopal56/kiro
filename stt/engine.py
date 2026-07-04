"""STT engine that exposes provider transcripts to downstream adapters."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from stt.base import STTProvider


class STTEngine:
    """Wraps an STT provider and exposes raw transcript strings."""

    def __init__(self, provider: STTProvider, language: Optional[str] = None) -> None:
        self.provider = provider
        self.language = language

    async def stream(self) -> AsyncIterator[str]:
        async for transcript in self.provider.stream():
            yield transcript

    async def close(self) -> None:
        await self.provider.close()