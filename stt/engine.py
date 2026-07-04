"""
STT engine: wraps a provider and emits InputEvent objects.

This is the only STT surface the agent core should import. Swapping
providers means constructing a different STTProvider; nothing downstream
changes.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from stt.schema import InputEvent, Modality  # adjust to your actual schema path
from stt.base import STTProvider


class STTEngine:
    """Adapts a provider's raw transcripts into InputEvent objects."""

    def __init__(self, provider: STTProvider, language: Optional[str] = None) -> None:
        self.provider = provider
        self.language = language

    async def stream(self) -> AsyncIterator[InputEvent]:
        async for transcript in self.provider.stream():
            yield InputEvent(
                text=transcript,
                modality=Modality.AUDIO,
                language=self.language,
            )

    async def close(self) -> None:
        await self.provider.close()