import os
import traceback
from dotenv import load_dotenv
from io import BytesIO
from pydub import AudioSegment
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from .logger import logger

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_TTS_ONLY_API_KEY"))

def tts_generate_audio(text: str, provider: str = "openai") -> AudioSegment:
    """
    Convert text to speech and return as AudioSegment.
    provider can be "openai" or "elevenlabs".
    """
    logger.info(f"Generating TTS audio for text: {text} using {provider}")

    try:
        if provider == "elevenlabs":
            audio = elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio)

        elif provider == "openai":
            response = openai_client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text
            )
            audio_bytes = response.read()

        else:
            raise ValueError("Invalid provider. Choose 'openai' or 'elevenlabs'.")

        logger.info(f"Generated TTS audio for text: {text} using {provider}")
        return AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")

    except Exception as e:
        logger.error(f"Failed to generate TTS audio with {provider}: {e}")
        logger.debug(traceback.format_exc())
        return None
