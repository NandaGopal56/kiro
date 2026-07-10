"""TTS engine that exposes provider speech synthesis."""

from __future__ import annotations

from tts.base import TTSProvider


class TTSEngine:
    """Simple wrapper around a TTS provider."""

    def __init__(self, provider: TTSProvider) -> None:
        self.provider = provider

    async def speak(self, text: str) -> None:
        await self.provider.speak(text)

    async def close(self) -> None:
        await self.provider.close()