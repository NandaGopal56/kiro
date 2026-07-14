"""Sarvam-backed text-to-speech provider."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv
import base64

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

        self.client = SarvamAI()
        self.model = model
        self.speaker = speaker
        self.language_code = language_code
        self.output_path = output_path
        self.last_output_path: Optional[Path] = None


    def _speak_sync(self, text: str) -> None:

        print(f'Request sent to TTS model for audio generation')
        response = self.client.text_to_speech.convert(
            text=text,
            target_language_code=self.language_code,
            model=self.model,
            speaker=self.speaker,
        )
        print(f'Audio generation completed, received audio data from TTS model')

        audio_bytes = base64.b64decode("".join(response.audios))

        return audio_bytes

    async def speak(self, text: str) -> str:
        if not text.strip():
            return ""
        audio_bytes = await asyncio.to_thread(self._speak_sync, text)
        return base64.b64encode(audio_bytes).decode("ascii")

    async def close(self) -> None:
        return None