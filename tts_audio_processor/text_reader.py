"""
Text reader module for the Agetic Conversation Bot.

This module contains the core agent implementations that process voice commands
and manage conversations.
"""
from typing import Dict, Any
from .tts_service import tts_generate_audio
from .audio_player import audio_queue
from .logger import logger


def on_llm_response(topic: str, payload: Dict[str, Any]) -> None:
    """Handle incoming LLM response and enqueue audio for playback."""
    try:
        logger.info(f"Processing LLM response: {payload}")
        text = payload.get("llm_response")
        if not text:
            return
        segment = tts_generate_audio(text)
        audio_queue.put(segment)
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}", exc_info=True)
