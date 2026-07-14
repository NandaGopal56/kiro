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
    """Plays base64-encoded audio, muting the mic for the duration of playback.

    `resume_delay` keeps the mic muted for a short window after playback
    finishes. pydub's `play()` returns as soon as the audio buffer is handed
    to the driver, not when the speaker has actually finished emitting sound,
    and there's also a brief acoustic tail (room echo/reverb) after that.
    Without this cooldown the mic reliably picks up the last chunk of the
    TTS output as if it were fresh input. Tune this per your hardware/room;
    200-400ms is a reasonable starting range.
    """

    def __init__(self, microphone: Optional[MicrophoneController] = None, resume_delay: float = 0.3):
        self.microphone = microphone
        self.resume_delay = resume_delay

    async def play_b64(self, audio_b64: str) -> None:
        if self.microphone is not None:
            self.microphone.pause()

        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            # pydub.playback.play() is blocking
            await asyncio.to_thread(play, audio)

            if self.resume_delay > 0:
                await asyncio.sleep(self.resume_delay)
        finally:
            if self.microphone is not None:
                self.microphone.resume()