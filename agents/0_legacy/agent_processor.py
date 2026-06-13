import asyncio
import logging

from communication_bus.inmemory_bus import InMemoryBus, bus as default_bus

from .video_topic_buffer import VideoTopicBuffer, video_buffer

logger = logging.getLogger(__name__)

class AgentProcessor:
    """Service for managing agent lifecycle and message handling."""

    name = "agents"

    def __init__(
        self,
        bus: InMemoryBus | None = None,
        video_buffer: VideoTopicBuffer | None = None,
    ):
        self._owns_bus = bus is None
        self.bus: InMemoryBus = bus or default_bus
        self.video_buffer: VideoTopicBuffer = video_buffer or globals()["video_buffer"]
        self._is_running = False
        self._run_task: asyncio.Task | None = None
        self._human_callback = None

    async def start(self, **kwargs) -> None:
        """Start the agent service asynchronously."""
        if self._is_running:
            logger.warning("Agent service already running")
            return

        try:
            from .communications.receiver import on_human_message

            logger.info("Starting Agent Service...")
            await self.bus.connect()
            self._human_callback = (
                lambda topic, payload: on_human_message(
                    topic,
                    payload,
                    response_bus=self.bus,
                )
            )
            self.bus.subscribe("voice/commands", self._human_callback)
            self.bus.subscribe("camera/front", self.video_buffer.on_frame)

            self._is_running = True
            logger.info("Agent Service started")

        except Exception as e:
            logger.error(f"Failed to start agent service: {e}", exc_info=True)
            self._is_running = False
            if self._owns_bus:
                await self.bus.disconnect()
            raise

    async def stop(self) -> None:
        """Stop the agent service and clean up resources."""
        if not self._is_running:
            return

        logger.info("Stopping Agent Service...")
        self._is_running = False
        if self._human_callback is not None:
            self.bus.unsubscribe("voice/commands", self._human_callback)
            self._human_callback = None
        self.bus.unsubscribe("camera/front", self.video_buffer.on_frame)

        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        if self._owns_bus:
            await self.bus.disconnect()
        logger.info("Agent Service stopped")


def create_service(
    bus: InMemoryBus | None = None,
    video_buffer: VideoTopicBuffer | None = None,
) -> AgentProcessor:
    return AgentProcessor(bus=bus, video_buffer=video_buffer)
