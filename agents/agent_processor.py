import asyncio
import logging
from communication_bus.inmemory_bus import InMemoryBus, bus
from .communications.receiver import on_human_message
from .VideoTopicBuffer import VideoTopicBuffer, video_buffer

logger = logging.getLogger(__name__)

class AgentProcessor:
    """Service for managing agent lifecycle and message handling."""

    def __init__(self):
        self.bus: InMemoryBus = bus
        self.video_buffer: VideoTopicBuffer = video_buffer
        self._is_running = False
        self._run_task: asyncio.Task | None = None

    async def start(self, **kwargs) -> None:
        """Start the agent service asynchronously."""
        if self._is_running:
            logger.warning("Agent service already running")
            return

        try:
            logger.info("Starting Agent Service...")
            await self.bus.connect()
            self.bus.subscribe("voice/commands", on_human_message)
            self.bus.subscribe("camera/front", self.video_buffer.on_frame)

            self._is_running = True
            logger.info("Agent Service started")

        except Exception as e:
            logger.error(f"Failed to start agent service: {e}", exc_info=True)
            self._is_running = False
            await self.bus.disconnect()
            raise

    async def stop(self) -> None:
        """Stop the agent service and clean up resources."""
        if not self._is_running:
            return

        logger.info("Stopping Agent Service...")
        self._is_running = False

        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        await self.bus.disconnect()
        logger.info("Agent Service stopped")