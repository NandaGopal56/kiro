from __future__ import annotations

import asyncio
from collections.abc import Callable


class FakeVoiceProcessor:
    """Small STT fake for tests and manual mock runs."""

    def __init__(
        self,
        utterances: list[str] | None = None,
        on_utterance: Callable[[str], None] | None = None,
    ) -> None:
        self.utterances = utterances or ["hello from mock stt"]
        self.on_utterance = on_utterance
        self._is_running = False

    async def run(self) -> None:
        self._is_running = True
        for utterance in self.utterances:
            if not self._is_running:
                break
            if self.on_utterance is not None:
                self.on_utterance(utterance)
            await asyncio.sleep(0)

    async def stop(self) -> None:
        self._is_running = False
