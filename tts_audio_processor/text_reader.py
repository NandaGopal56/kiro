"""
Text reader module for the Agetic Conversation Bot.

This module contains the core agent implementations that process voice commands
and manage conversations.
"""
from typing import Dict, Any

from .logger import logger


def extract_response_text(payload: Dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return payload
    return str(payload.get("text") or payload.get("llm_response") or "")


def handle_llm_response(payload: Dict[str, Any] | str, synthesizer, player) -> None:
    text = extract_response_text(payload)
    if not text:
        return
    segment = synthesizer(text)
    if segment is not None:
        player.enqueue(segment)


def _default_synthesizer(text: str):
    from .tts_service import tts_generate_audio

    return tts_generate_audio(text)


def make_llm_response_handler(synthesizer=None, player=None):
    synthesizer = synthesizer or _default_synthesizer
    if player is None:
        from .audio_player import TTSPlayer

        player = TTSPlayer()
        player.start()

    def _handler(topic: str, payload: Dict[str, Any] | str) -> None:
        try:
            logger.info(f"Processing LLM response: {payload}")
            handle_llm_response(payload, synthesizer, player)
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}", exc_info=True)

    return _handler


def on_llm_response(topic: str, payload: Dict[str, Any]) -> None:
    """Handle incoming LLM response and enqueue audio for playback."""
    try:
        from .audio_player import audio_queue

        text = extract_response_text(payload)
        if text:
            audio_queue.put(_default_synthesizer(text))
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}", exc_info=True)
