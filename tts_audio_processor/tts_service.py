import os
import traceback
from io import BytesIO

from dotenv import load_dotenv

from .logger import logger

load_dotenv()

_openai_client = None
_elevenlabs_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_elevenlabs_client():
    global _elevenlabs_client
    if _elevenlabs_client is None:
        from elevenlabs.client import ElevenLabs

        _elevenlabs_client = ElevenLabs(
            api_key=os.getenv("ELEVEN_LABS_TTS_ONLY_API_KEY")
        )
    return _elevenlabs_client

def tts_generate_audio(text: str, provider: str = "openai"):
    """
    Convert text to speech and return as AudioSegment.
    provider can be "openai" or "elevenlabs".
    """
    logger.info(f"Generating TTS audio for text: {text} using {provider}")

    try:
        if provider == "elevenlabs":
            audio = _get_elevenlabs_client().text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio)

        elif provider == "openai":
            response = _get_openai_client().audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text
            )
            audio_bytes = response.read()

        else:
            raise ValueError("Invalid provider. Choose 'openai' or 'elevenlabs'.")

        from pydub import AudioSegment

        logger.info(f"Generated TTS audio for text: {text} using {provider}")
        return AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")

    except Exception as e:
        logger.error(f"Failed to generate TTS audio with {provider}: {e}")
        logger.debug(traceback.format_exc())
        return None
