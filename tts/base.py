"""Abstract base for text-to-speech providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """Common interface all TTS providers must implement."""

    @abstractmethod
    async def speak(self, text: str) -> str:
        """Convert text to speech and return base64-encoded audio bytes."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
        ...