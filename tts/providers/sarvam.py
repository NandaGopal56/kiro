"""Sarvam-backed text-to-speech provider."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv
import base64
import io

from pydub import AudioSegment
from pydub.playback import play

load_dotenv(find_dotenv()) 

from sarvamai import SarvamAI

from tts.base import TTSProvider


class SarvamTTS(TTSProvider):
    """Synthesizes speech through Sarvam AI and writes audio to disk."""

    def __init__(
        self,
        model: str = "bulbul:v3",
        speaker: str = "shubh",
        language_code: str = "en-IN",
        output_path: Optional[str] = None,
    ) -> None:
        if not os.environ.get("SARVAM_API_KEY"):
            raise ValueError("SARVAM_API_KEY is required for the Sarvam TTS provider")

        self.client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))

        self.model = model
        self.speaker = speaker
        self.language_code = language_code
        self.output_path = output_path
        self.last_output_path: Optional[Path] = None


    def _speak_sync(self, text: str) -> None:

        response = self.client.text_to_speech.convert(
            text=text,
            target_language_code=self.language_code,
            model=self.model,
            speaker=self.speaker,
        )

        audio_bytes = base64.b64decode("".join(response.audios))

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        play(audio)

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        await asyncio.to_thread(self._speak_sync, text)

    async def close(self) -> None:
        return None