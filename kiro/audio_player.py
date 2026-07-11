# audio_player.py

from __future__ import annotations

import asyncio
import base64
import io
from typing import Optional, Protocol

from pydub import AudioSegment
from pydub.playback import play


class MicrophoneController(Protocol):
    """Anything that can be paused/resumed around playback (e.g. KiroMicrophone)."""

    def pause(self) -> None: ...
    def resume(self) -> None: ...


class AudioPlayer:
    """Plays base64-encoded audio, muting the mic for the duration of playback."""

    def __init__(self, microphone: Optional[MicrophoneController] = None):
        self.microphone = microphone

    async def play_b64(self, audio_b64: str) -> None:
        if self.microphone is not None:
            self.microphone.pause()

        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            # pydub.playback.play() is blocking
            await asyncio.to_thread(play, audio)
        finally:
            if self.microphone is not None:
                self.microphone.resume()