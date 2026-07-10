# audio_player.py

from __future__ import annotations

import asyncio
import base64
import io
from typing import Optional, Protocol

from pydub import AudioSegment
from pydub.playback import play


class MicrophoneController(Protocol):
    async def pause(self) -> None: ...
    async def resume(self) -> None: ...


class AudioPlayer:
    def __init__(self, microphone: Optional[MicrophoneController] = None):
        self.microphone = microphone

    async def play_b64(self, audio_b64: str) -> None:
        # if self.microphone:
        #     await self.microphone.pause()

        try:
            audio_bytes = base64.b64decode(audio_b64)

            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format="wav",
            )

            # pydub.playback.play() is blocking
            await asyncio.to_thread(play, audio)

        finally:
            pass
            # if self.microphone:
            #     await self.microphone.resume()