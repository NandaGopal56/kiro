"""Abstract base for text-to-speech providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """Common interface all TTS providers must implement."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Convert text to speech."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
        ...