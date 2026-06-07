"""
Agents module for the Agetic Conversation Bot.

This module contains the core agent implementations that process human messages
and manage conversations.
"""
import logging
import asyncio
from ..bot import invoke_conversation

logger = logging.getLogger(__name__)

def on_human_message(topic: str, message: str) -> None:
    """Handle incoming human messages.
    
    Args:
        topic: The topic on which the message was received
        message: The text of the human message
    """
    try:
        logger.info(f"Processing human message on Topic: {topic}, message: {message}")

        asyncio.create_task(invoke_conversation(message, thread_id=1))
    
    except Exception as e:
        logger.error(f"Error processing human message: {e}", exc_info=True)
