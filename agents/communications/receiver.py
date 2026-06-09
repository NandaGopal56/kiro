"""
Agents module for the Agetic Conversation Bot.

This module contains the core agent implementations that process human messages
and manage conversations.
"""
import logging
import asyncio
from typing import Any

from ..bot import invoke_conversation

logger = logging.getLogger(__name__)


def extract_message_text(payload: Any) -> str:
    """Accept canonical dict payloads and legacy plain strings."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return str(payload.get("text") or payload.get("message") or "")
    return str(payload or "")


async def _drain_conversation(
    message: str,
    thread_id: int = 1,
    response_bus=None,
) -> None:
    """Run the streaming conversation generator from a background task."""
    chunks = []
    async for chunk in invoke_conversation(message, thread_id=thread_id):
        chunks.append(chunk)

    if response_bus is not None and chunks:
        await response_bus.publish(
            "voice/commands/llm_response",
            {"text": "".join(chunks)},
        )


def _log_task_failure(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"Error processing human message task: {exc}", exc_info=True)


def on_human_message(topic: str, message: Any, response_bus=None) -> None:
    """Handle incoming human messages.
    
    Args:
        topic: The topic on which the message was received
        message: The text of the human message
    """
    try:
        text = extract_message_text(message)
        if not text:
            logger.warning("Ignoring empty human message on topic %s", topic)
            return

        logger.info(f"Processing human message on Topic: {topic}, message: {text}")

        task = asyncio.get_running_loop().create_task(
            _drain_conversation(text, thread_id=1, response_bus=response_bus)
        )
        task.add_done_callback(_log_task_failure)
    
    except Exception as e:
        logger.error(f"Error processing human message: {e}", exc_info=True)
