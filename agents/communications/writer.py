
from communication_bus.inmemory_bus import bus
from typing import Dict, Any
import logging
import traceback

# Initialize logger
logger = logging.getLogger(__name__)

async def write_response_to_bus(payload: Dict[str, Any]):
    '''
    This function is used to write the response to the bus
    '''
    try:
        logger.debug(f"Writing response to bus: {payload}")
        await bus.connect()
        logger.debug(f"Connected to bus")
        await bus.publish("voice/commands/llm_response", payload)
        logger.debug(f"Published response to bus")
    except Exception as e:
        logger.error(f"Error writing response to bus: {e}", exc_info=True)
        logger.debug(traceback.format_exc())
